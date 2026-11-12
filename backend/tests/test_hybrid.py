from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.idp_engine.hybrid import (
    HybridPipeline,
    assess_quality,
    reconcile,
)
from backend.app.idp_engine.local_vlm import LocalVLM
from backend.app.schemas.invoice import ExtractionResponse, InvoiceData, LineItem, MathCheck
from backend.app.main import app


# ---------------------------------------------------------------- helpers ---

def _png() -> bytes:
    image = np.full((240, 180, 3), 255, dtype=np.uint8)
    cv2.putText(image, "receipt", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _invoice(**overrides) -> InvoiceData:
    base = dict(
        seller="CỬA HÀNG ABC",
        invoice_id="HD-001",
        total_cost=66000.0,
        subtotal=60000.0,
        tax=6000.0,
        timestamp=datetime(2026, 10, 22, 14, 30),
        items=[LineItem(name="Cà phê", quantity=2, unit_price=30000, amount=60000, confidence=0.9)],
    )
    base.update(overrides)
    return InvoiceData(**base)


def _details(invoice: InvoiceData, conf: float = 0.9) -> dict[str, dict]:
    detail = {"kie_conf": conf, "ocr_conf": conf, "method": "labeled_inline", "box_ids": [1]}
    names = ("seller", "invoice_id", "total_cost", "subtotal", "tax", "timestamp")
    return {name: {**detail, "value": getattr(invoice, name)} for name in names if getattr(invoice, name, None)}


def _response(invoice: InvoiceData, *, status="VALIDATED", confidence=0.9, passed=True, details=None, aligned=None):
    return ExtractionResponse(
        invoice=invoice,
        status=status,
        confidence=confidence,
        strategy="2-stage-paddleocr-vietocr-kie",
        warnings=["warn"] if status != "VALIDATED" else [],
        math=MathCheck(line_sum=60000, tax=6000, expected_total=66000, actual_total=invoice.total_cost or 0,
                       delta=0, passed=passed),
        timings={"preprocess_ms": 300, "ocr_ms": 250, "kie_ms": 25, "total_ms": 600},
        engine={"backend": "paddleocr", "device": "gpu"},
        field_details=details or {},
        aligned_image_url=aligned,
    )


class StubBase:
    strategy = "2-stage-paddleocr-vietocr-kie"

    def __init__(self, response: ExtractionResponse):
        self.response = response

    def run(self, *args, **kwargs):
        return self.response


class StubVLM:
    def __init__(self, data: InvoiceData | None = None, evidence: dict | None = None,
                 available: bool = True, provider: str = "local", fail: bool = False):
        self.provider = provider
        self.api_key = "x"
        self.model = "stub"
        self._local = None
        self._data = data
        self._evidence = evidence or {}
        self._available = available
        self._fail = fail
        self.calls = 0

    def fallback_available(self):
        return (True, "") if self._available else (False, "stub unavailable")

    def extract(self, path):
        self.calls += 1
        if self._fail:
            raise RuntimeError("stub VLM error")
        payload = (self._data or InvoiceData()).model_dump(mode="json")
        payload["evidence"] = self._evidence
        return self._data or InvoiceData(), [], json.dumps(payload, ensure_ascii=False)


# ------------------------------------------------------------- assessment ---

def test_assess_quality_passes_clean_result():
    invoice = _invoice()
    result = assess_quality(invoice, 0.95, True, 0.85, _details(invoice))
    assert result.needs_vlm is False
    assert result.triggers == []


def test_assess_quality_triggers_on_missing_critical_field():
    invoice = _invoice(total_cost=0.0)
    details = _details(invoice)
    details.pop("total_cost", None)
    result = assess_quality(invoice, 0.95, True, 0.85, details)
    assert result.needs_vlm
    assert any("total_cost" in trigger for trigger in result.triggers)


def test_assess_quality_triggers_on_low_kie_confidence():
    invoice = _invoice()
    details = _details(invoice)
    details["seller"]["kie_conf"] = 0.2
    result = assess_quality(invoice, 0.95, True, 0.85, details)
    assert result.needs_vlm
    assert any("seller" in trigger for trigger in result.triggers)


def test_assess_quality_triggers_on_failed_validation():
    invoice = _invoice(total_cost=99000.0)
    result = assess_quality(invoice, 0.95, False, 0.85, _details(invoice))
    assert result.needs_vlm
    assert any("Validation" in trigger for trigger in result.triggers)


# ----------------------------------------------------------- reconciliation ---

def test_reconcile_never_overwrites_strong_kie():
    kie = _invoice(seller="CỬA HÀNG ABC")
    vlm = InvoiceData(seller="CỬA HÀNG XYZ")
    result, patch = reconcile(kie, _details(kie), vlm)
    assert result.data.seller == "CỬA HÀNG ABC"
    assert result.field_sources["seller"] == "kie"
    assert result.conflicts and result.conflicts[0]["resolved"] is True
    assert result.conflicts[0]["chosen_source"] == "kie"


def test_reconcile_adopts_vlm_for_weak_kie_with_evidence():
    kie = _invoice(total_cost=66000.0)
    details = _details(kie)
    details["total_cost"]["kie_conf"] = 0.3
    vlm = InvoiceData(total_cost=56000.0)
    result, patch = reconcile(kie, details, vlm, {"total_cost": "Tổng cộng: 56.000"})
    assert result.data.total_cost == 56000.0
    assert result.field_sources["total_cost"] == "vlm"
    assert result.unresolved, "conflict must stay unresolved → NEEDS_REVIEW"
    assert "total_cost" in result.vlm_accepted
    # No fabricated confidence for VLM-sourced fields.
    assert "total_cost" not in result.data.field_confidence
    assert patch["total_cost"]["kie_conf"] is None
    assert patch["total_cost"]["evidence"] == "Tổng cộng: 56.000"


def test_reconcile_fills_missing_field_from_vlm():
    kie = _invoice(invoice_id=None)
    details = _details(kie)
    details.pop("invoice_id", None)
    vlm = InvoiceData(invoice_id="HD-VLM-9")
    result, _ = reconcile(kie, details, vlm, {"invoice_id": "Số: HD-VLM-9"})
    assert result.data.invoice_id == "HD-VLM-9"
    assert result.field_sources["invoice_id"] == "vlm"


def test_reconcile_rejects_vlm_without_evidence():
    kie = InvoiceData()
    vlm = InvoiceData(seller="CỬA HÀNG BỊA")
    result, _ = reconcile(kie, {}, vlm)  # no evidence dict
    assert result.data.seller is None
    assert result.field_sources["seller"] == "none"
    assert result.vlm_rejected


def test_reconcile_agreement_marks_kie_vlm():
    kie = _invoice(seller="Cửa hàng ABC")
    vlm = InvoiceData(seller="CỬA HÀNG ABC")
    result, _ = reconcile(kie, _details(kie), vlm)
    assert result.field_sources["seller"] == "kie+vlm"
    assert not result.conflicts


def test_reconcile_keeps_kie_items_and_fills_empty_grid():
    kie = _invoice()
    vlm_extra = InvoiceData(items=[LineItem(name="Bánh", quantity=1, unit_price=9000, amount=9000)])
    result, _ = reconcile(kie, _details(kie), vlm_extra)
    assert result.field_sources["items"] == "kie"
    assert [item.name for item in result.data.items] == ["Cà phê"]

    empty = _invoice(items=[])
    details = _details(empty)
    details.pop("items", None)
    result, patch = reconcile(empty, details, vlm_extra)
    assert result.field_sources["items"] == "vlm"
    assert result.data.items and result.data.items[0].name == "Bánh"
    assert patch["items"]["source"] == "vlm"


def test_reconcile_rejects_implausible_vlm_items():
    empty = _invoice(items=[])
    vlm_bad = InvoiceData(items=[LineItem(name="12345", quantity=1, unit_price=1, amount=1)])
    result, _ = reconcile(empty, {}, vlm_bad)
    assert result.field_sources["items"] == "none"
    assert not result.data.items


# --------------------------------------------------------------- local VLM ---

def test_local_vlm_parse_output_with_evidence():
    raw = json.dumps({"seller": "ABC", "total_cost": 66000, "items": [], "evidence": {"seller": "ABC"}})
    data, evidence = LocalVLM.parse_output(raw)
    assert data.seller == "ABC"
    assert evidence == {"seller": "ABC"}


def test_local_vlm_parse_fenced_json():
    raw = "```json\n{\"total_cost\": 1}\n```"
    data, _ = LocalVLM.parse_output(raw)
    assert data.total_cost == 1.0


def test_local_vlm_parse_rejects_garbage():
    with pytest.raises(ValueError):
        LocalVLM.parse_output("không có json ở đây")


# ---------------------------------------------------------------- pipeline ---

def test_hybrid_skips_vlm_when_kie_is_reliable(tmp_path, monkeypatch):
    from backend.app.config import get_settings

    settings = get_settings()
    invoice = _invoice()
    base = StubBase(_response(invoice, details=_details(invoice)))
    vlm = StubVLM(InvoiceData(seller="KHÁC"))
    pipeline = HybridPipeline(base=base, vlm=vlm)
    response = pipeline.run(Path("dummy.png"))
    assert response.vlm_triggered is False
    assert vlm.calls == 0
    assert response.strategy == "hybrid-paddleocr-kie-vlm"
    assert response.sources.get("seller") == "kie"
    assert response.status == "VALIDATED"


def test_hybrid_runs_vlm_and_fills_missing_total(tmp_path, monkeypatch):
    from backend.app.config import get_settings

    settings = get_settings()
    aligned = settings.processed_dir / "aligned_stub.png"
    aligned.write_bytes(_png())
    invoice = _invoice(total_cost=0.0, subtotal=0.0, tax=0.0, items=[])
    details = _details(invoice)
    details.pop("total_cost", None)
    base = StubBase(_response(invoice, status="NEEDS_REVIEW", passed=False,
                              details=details, aligned="/processed/aligned_stub.png"))
    vlm = StubVLM(
        InvoiceData(total_cost=66000.0, timestamp=datetime(2026, 10, 22, 14, 30)),
        evidence={"total_cost": "Tổng cộng: 66.000", "timestamp": "22/10/2026 14:30"},
    )
    pipeline = HybridPipeline(base=base, vlm=vlm)
    response = pipeline.run(Path("dummy.png"))
    assert response.vlm_triggered is True
    assert vlm.calls == 1
    assert response.invoice.total_cost == 66000.0
    assert response.sources["total_cost"] == "vlm"
    assert "vlm_ms" in response.timings
    assert response.fallback_reasons
    assert response.status == "NEEDS_REVIEW"


def test_hybrid_conflict_forces_needs_review(tmp_path, monkeypatch):
    from backend.app.config import get_settings

    settings = get_settings()
    aligned = settings.processed_dir / "aligned_stub2.png"
    aligned.write_bytes(_png())
    invoice = _invoice(total_cost=66000.0)
    details = _details(invoice)
    details["total_cost"]["kie_conf"] = 0.3
    base = StubBase(_response(invoice, details=details, aligned="/processed/aligned_stub2.png"))
    vlm = StubVLM(InvoiceData(total_cost=56000.0), evidence={"total_cost": "Total: 56.000"})
    pipeline = HybridPipeline(base=base, vlm=vlm)
    response = pipeline.run(Path("dummy.png"))
    assert response.invoice.total_cost == 56000.0
    assert response.conflicts
    assert any(not conflict["resolved"] for conflict in response.conflicts)
    assert response.status == "NEEDS_REVIEW"
    assert any("xung đột" in warning.lower() for warning in response.warnings)


def test_hybrid_degrades_when_vlm_unavailable(tmp_path, monkeypatch):
    invoice = _invoice(total_cost=0.0, items=[])
    details = _details(invoice)
    details.pop("total_cost", None)
    base = StubBase(_response(invoice, status="NEEDS_REVIEW", passed=False, details=details))
    vlm = StubVLM(InvoiceData(), available=False)
    pipeline = HybridPipeline(base=base, vlm=vlm)
    response = pipeline.run(Path("dummy.png"))
    assert response.vlm_triggered is False
    assert vlm.calls == 0
    assert any("chưa khả dụng" in warning for warning in response.warnings)
    assert response.invoice is not None


def test_hybrid_vlm_error_keeps_kie_result(tmp_path, monkeypatch):
    from backend.app.config import get_settings

    settings = get_settings()
    aligned = settings.processed_dir / "aligned_stub3.png"
    aligned.write_bytes(_png())
    invoice = _invoice()
    details = _details(invoice)
    details["seller"]["kie_conf"] = 0.2
    base = StubBase(_response(invoice, status="NEEDS_REVIEW", details=details,
                              aligned="/processed/aligned_stub3.png"))
    vlm = StubVLM(InvoiceData(), fail=True)
    pipeline = HybridPipeline(base=base, vlm=vlm)
    response = pipeline.run(Path("dummy.png"))
    assert response.vlm_triggered is False
    assert any("VLM dự phòng lỗi" in warning for warning in response.warnings)
    assert response.invoice.seller == "CỬA HÀNG ABC"


# --------------------------------------------------------------------- API ---

def test_extract_hybrid_mode_returns_provenance():
    client = TestClient(app)
    text = "\n".join([
        "CỬA HÀNG ABC",
        "Số hóa đơn: HD-HYB-01",
        "Cà phê 2 30.000 60.000",
        "Tổng cộng: 60.000",
    ])
    response = client.post(
        "/api/extract",
        files={"file": ("receipt.png", _png(), "image/png")},
        data={"mode": "hybrid", "ocr_text": text},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategy"] == "hybrid-paddleocr-kie-vlm"
    assert body["vlm_triggered"] is False  # mock provider in tests
    assert isinstance(body["sources"], dict)
    assert isinstance(body["fallback_reasons"], list)
    assert isinstance(body["conflicts"], list)


def test_extract_ocr_kie_alias_mode():
    client = TestClient(app)
    text = "Tổng cộng: 60.000"
    response = client.post(
        "/api/extract",
        files={"file": ("receipt.png", _png(), "image/png")},
        data={"mode": "ocr_kie", "ocr_text": text},
    )
    assert response.status_code == 200, response.text
    assert "2-stage" in response.json()["strategy"]


def test_ocr_status_reports_vlm_fields():
    client = TestClient(app)
    response = client.get("/api/ocr/status")
    assert response.status_code == 200
    body = response.json()
    assert "vlm_provider" in body
    assert "vlm_available" in body
