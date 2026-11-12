"""Hybrid pipeline quality gates and OCR-KIE ↔ VLM reconciliation.

Two pure-logic halves:

1. **Assess** — decide whether the KIE result is good enough to skip the VLM.
   Triggers stay conservative: the VLM runs only when a critical field is
   missing/low-confidence, validation fails, or the goods grid looks collapsed.
   A clean KIE result must not pay the VLM latency.
2. **Reconcile** — merge KIE-first with VLM-fills-gaps. KIE values with solid
   evidence are never overwritten just because the VLM disagrees; VLM values
   are only accepted for fields where KIE had nothing solid. Unresolved
   conflicts surface as warnings and force ``NEEDS_REVIEW``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any

from ..schemas.invoice import InvoiceData, LineItem

# Fields whose absence most likely means the extraction missed the receipt's
# header or summary block. Present + confident ⇒ skip the VLM.
CRITICAL_FIELDS = ("total_cost", "seller", "invoice_id", "timestamp")

TEXT_FIELDS = (
    "invoice_id", "invoice_series", "invoice_form", "seller", "buyer",
    "address", "buyer_address", "tax_id", "buyer_tax_id", "currency",
)
MONEY_FIELDS = ("subtotal", "tax", "discount", "total_cost")

# kie_conf >= this ⇒ KIE evidence is strong enough that the VLM must not
# overwrite it (spec: never overwrite KIE solely because the VLM disagrees).
STRONG_KIE_CONF = 0.75
# VLM-proposed values below this confidence are never accepted (no evidence).
MIN_VLM_ACCEPT_CONF = 0.3
# Money fields: a VLM/KIE numeric delta inside this ratio is a punctuation
# misread, not a real conflict.
MONEY_DELTA_RATIO = 0.02
# Minimal quality bar for item rows accepted from the VLM.
_MIN_ITEM_NAME_LEN = 1
_MIN_ITEM_MONEY = 1.0


@dataclass(slots=True)
class QualityAssessment:
    """Result of the gate that decides whether VLM fallback is worth running."""

    needs_vlm: bool
    triggers: list[str] = dc_field(default_factory=list)
    critical_missing: list[str] = dc_field(default_factory=list)

    def describe(self) -> str:
        return "; ".join(self.triggers) if self.triggers else "KIE đạt ngưỡng chất lượng"


def assess_quality(
    data: InvoiceData,
    confidence: float,
    validation_passed: bool,
    threshold: float,
    field_details: dict[str, dict[str, Any]] | None = None,
) -> QualityAssessment:
    """Decide whether the OCR→KIE result justifies a VLM fallback run."""
    triggers: list[str] = []
    details = field_details or {}

    critical_missing = [
        name for name in CRITICAL_FIELDS
        if not _field_has_value(data, name) or details.get(name, {}).get("kie_conf", 0) < 0.5
    ]
    if critical_missing:
        triggers.append("Thiếu trường quan trọng: " + ", ".join(critical_missing))

    weak_critical = [
        name for name in CRITICAL_FIELDS
        if name not in critical_missing
        and _field_has_value(data, name)
        and details.get(name, {}).get("kie_conf", 0) < threshold - 0.1
    ]
    if weak_critical:
        triggers.append("Trường quan trọng có độ tin cậy KIE thấp: " + ", ".join(weak_critical))

    if not validation_passed:
        triggers.append("Validation tổng tiền không khớp (hàng hóa + thuế − giảm giá ≠ tổng).")
    if confidence < threshold:
        triggers.append(f"Độ tin cậy OCR tổng thể {confidence:.2f} < ngưỡng {threshold:.2f}.")
    if not data.items:
        triggers.append("Không rút trích được dòng hàng hóa nào.")
    elif len(data.items) == 1 and (data.total_cost or 0) > 0:
        item_amount = data.items[0].amount or data.items[0].quantity * data.items[0].unit_price
        if item_amount and abs(item_amount - data.total_cost) <= max(1.0, item_amount * 0.05):
            # One line equal to the total is suspicious: a multi-line invoice
            # whose grid collapsed into one row — OR a genuinely single-item
            # receipt. The VLM gets a chance to find missed rows.
            triggers.append("Chỉ có 1 dòng hàng khớp tổng — nghi ngờ gộp dòng.")

    return QualityAssessment(
        needs_vlm=bool(triggers), triggers=triggers, critical_missing=critical_missing
    )


def _field_has_value(data: InvoiceData, name: str) -> bool:
    value = getattr(data, name, None)
    if name in MONEY_FIELDS:
        return bool(value and float(value) > 0)
    return bool(value and str(value).strip())


def _strong(detail: dict[str, Any] | None) -> bool:
    return bool(detail and detail.get("kie_conf", 0.0) >= STRONG_KIE_CONF)


def _numbers_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, max(abs(a), abs(b)) * MONEY_DELTA_RATIO)


def _text_equivalent(a: Any, b: Any) -> bool:
    """Lenient string compare: casefold + drop non-alphanumerics (both sides
    are already Unicode-normalized by _normalize_invoice before comparing)."""
    def fold(value: Any) -> str:
        if isinstance(value, datetime):
            value = value.isoformat(timespec="seconds")
        return re.sub(r"[^0-9a-zà-ỹ]+", "", str(value or "").casefold())

    return fold(a) != "" and fold(a) == fold(b)


def _money_equal(a: Any, b: Any) -> bool:
    try:
        return _numbers_close(float(a or 0), float(b or 0))
    except (TypeError, ValueError):
        return False


def _datetime_equal(a: Any, b: Any) -> bool:
    a_dt = a if isinstance(a, datetime) else None
    b_dt = b if isinstance(b, datetime) else None
    if a_dt is None or b_dt is None:
        return False
    return a_dt == b_dt


def _items_equivalent(a: list[LineItem], b: list[LineItem]) -> bool:
    if len(a) != len(b):
        return False
    for item_a, item_b in zip(a, b):
        if not _text_equivalent(item_a.name, item_b.name):
            return False
        if not _money_equal(
            item_a.amount or item_a.quantity * item_a.unit_price,
            item_b.amount or item_b.quantity * item_b.unit_price,
        ):
            return False
    return True


def _item_vlm_plausible(item: LineItem) -> bool:
    """Reject VLM item rows that look fabricated: no name or impossible money."""
    name = (item.name or "").strip()
    if len(name) < _MIN_ITEM_NAME_LEN or not re.search(r"[^\d\s]", name):
        return False
    amount = item.amount or item.quantity * item.unit_price
    try:
        if float(amount) < _MIN_ITEM_MONEY or float(amount) > 1e13:
            return False
    except (TypeError, ValueError):
        return False
    return True


@dataclass(slots=True)
class ReconciliationResult:
    """Outcome of merging the KIE draft with the VLM draft."""

    data: InvoiceData
    field_sources: dict[str, str] = dc_field(default_factory=dict)
    conflicts: list[dict[str, Any]] = dc_field(default_factory=list)
    vlm_accepted: list[str] = dc_field(default_factory=list)
    vlm_rejected: list[dict[str, str]] = dc_field(default_factory=list)

    @property
    def unresolved(self) -> list[dict[str, Any]]:
        return [item for item in self.conflicts if not item.get("resolved", True)]


def reconcile(
    kie: InvoiceData,
    kie_details: dict[str, dict[str, Any]],
    vlm: InvoiceData,
    vlm_evidence: dict[str, str] | None = None,
    vlm_field_confidence: dict[str, float] | None = None,
) -> tuple[ReconciliationResult, dict[str, dict[str, Any]]]:
    """KIE-first merge. Returns the reconciled invoice plus full provenance.

    Rules per field (text/money/datetime):
    - KIE present & strong (kie_conf >= STRONG_KIE_CONF): keep KIE. VLM
      disagreement is recorded as a *resolved* conflict (policy = trust KIE).
    - KIE present & weak, VLM agrees: keep KIE, source ``kie+vlm``.
    - KIE present & weak, VLM disagrees with evidence: adopt VLM, mark the
      conflict *unresolved* (human must confirm) → NEEDS_REVIEW downstream.
    - KIE missing, VLM present: adopt VLM, source ``vlm`` (gap fill).
    - VLM value missing/None: keep whatever KIE had.

    Items: KIE rows win whenever present; VLM rows only fill an empty grid,
    and only rows that pass a plausibility check (non-numeric name, sane
    amounts).
    """
    vlm_evidence = vlm_evidence or {}
    vlm_conf = vlm_field_confidence or {}
    chosen_payload = kie.model_dump()
    sources: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    details_patch: dict[str, dict[str, Any]] = {}

    def adopt_vlm(name: str, value: Any) -> None:
        chosen_payload[name] = value
        sources[name] = "vlm"
        accepted.append(name)
        detail = {
            "value": value.isoformat(timespec="seconds") if isinstance(value, datetime) else value,
            "kie_conf": None,
            "ocr_conf": None,
            "method": "vlm_local",
            "box_ids": [],
            "source": "vlm",
        }
        evidence = vlm_evidence.get(name)
        if evidence:
            detail["evidence"] = evidence
        vlm_score = vlm_conf.get(name)
        if vlm_score is not None:
            detail["vlm_conf"] = round(float(vlm_score), 3)
        details_patch[name] = detail

    def keep_kie(name: str, source: str = "kie") -> None:
        sources[name] = source
        detail = kie_details.get(name)
        if detail is not None:
            merged = dict(detail)
            merged.setdefault("source", source)
            details_patch[name] = merged

    for name in TEXT_FIELDS:
        kie_value = getattr(kie, name, None)
        vlm_value = getattr(vlm, name, None)
        kie_present = bool(kie_value and str(kie_value).strip())
        vlm_present = bool(vlm_value and str(vlm_value).strip())
        if kie_present and vlm_present:
            same = _text_equivalent(kie_value, vlm_value)
            if same:
                keep_kie(name, "kie+vlm")
            elif _strong(kie_details.get(name)):
                keep_kie(name, "kie")
                conflicts.append(_conflict(name, kie_value, vlm_value, kie_value, "kie", True,
                                           "KIE đủ tin cậy — giữ KIE, ghi nhận bất đồng VLM"))
            else:
                evidence = vlm_evidence.get(name)
                if evidence:
                    adopt_vlm(name, vlm_value)
                    conflicts.append(_conflict(name, kie_value, vlm_value, vlm_value, "vlm", False,
                                               f"KIE yếu + VLM có bằng chứng → dùng VLM (bằng chứng: “{evidence}”)"))
                else:
                    keep_kie(name, "kie")
                    rejected.append({"field": name, "reason": "VLM không có bằng chứng"})
        elif kie_present:
            keep_kie(name, "kie")
        elif vlm_present:
            evidence = vlm_evidence.get(name)
            if evidence:
                adopt_vlm(name, vlm_value)
            else:
                sources[name] = "none"
                rejected.append({"field": name, "reason": "VLM thiếu bằng chứng cho trường trống ở KIE"})
        else:
            sources[name] = "none"

    for name in MONEY_FIELDS:
        kie_value = float(getattr(kie, name, 0.0) or 0.0)
        vlm_value = float(getattr(vlm, name, 0.0) or 0.0)
        kie_present = kie_value > 0
        vlm_present = vlm_value > 0
        if kie_present and vlm_present:
            if _money_equal(kie_value, vlm_value):
                keep_kie(name, "kie+vlm")
            elif _strong(kie_details.get(name)):
                keep_kie(name, "kie")
                conflicts.append(_conflict(name, kie_value, vlm_value, kie_value, "kie", True,
                                           "Số tiền KIE đủ tin cậy — giữ KIE"))
            elif vlm_conf.get(name, 1.0) < MIN_VLM_ACCEPT_CONF:
                keep_kie(name, "kie")
                rejected.append({"field": name, "reason": "VLM tin cậy quá thấp"})
            else:
                adopt_vlm(name, vlm_value)
                conflicts.append(_conflict(name, kie_value, vlm_value, vlm_value, "vlm", False,
                                           "KIE yếu + VLM khác số tiền → dùng VLM, cần người duyệt"))
        elif kie_present:
            keep_kie(name, "kie")
        elif vlm_present:
            if vlm_conf.get(name, 1.0) < MIN_VLM_ACCEPT_CONF:
                sources[name] = "none"
                rejected.append({"field": name, "reason": "VLM tin cậy quá thấp cho trường trống"})
            else:
                adopt_vlm(name, vlm_value)
        else:
            sources[name] = "none"

    kie_ts = getattr(kie, "timestamp", None)
    vlm_ts = getattr(vlm, "timestamp", None)
    if kie_ts is not None and vlm_ts is not None:
        if _datetime_equal(kie_ts, vlm_ts):
            keep_kie("timestamp", "kie+vlm")
        elif _strong(kie_details.get("timestamp")):
            keep_kie("timestamp", "kie")
            conflicts.append(_conflict("timestamp", kie_ts.isoformat(), vlm_ts.isoformat(),
                                       kie_ts.isoformat(), "kie", True, "Thời gian KIE đủ tin cậy — giữ KIE"))
        elif vlm_evidence.get("timestamp"):
            adopt_vlm("timestamp", vlm_ts)
            conflicts.append(_conflict("timestamp", kie_ts.isoformat(), vlm_ts.isoformat(),
                                       vlm_ts.isoformat(), "vlm", False,
                                       "KIE yếu + VLM có bằng chứng → dùng VLM, cần người duyệt"))
        else:
            keep_kie("timestamp", "kie")
            rejected.append({"field": "timestamp", "reason": "VLM thiếu bằng chứng"})
    elif kie_ts is not None:
        keep_kie("timestamp", "kie")
    elif vlm_ts is not None and vlm_evidence.get("timestamp"):
        adopt_vlm("timestamp", vlm_ts)
    else:
        sources["timestamp"] = "none"

    # Items: KIE grid wins; VLM only fills an empty grid with plausible rows.
    kie_items = list(kie.items or [])
    vlm_items = [item for item in (vlm.items or []) if _item_vlm_plausible(item)]
    if kie_items:
        sources["items"] = "kie"
        detail = kie_details.get("items")
        if detail is not None:
            details_patch["items"] = {**detail, "source": "kie"}
        if vlm_items and not _items_equivalent(kie_items, vlm_items):
            conflicts.append(_conflict("items", f"{len(kie_items)} dòng (KIE)",
                                       f"{len(vlm_items)} dòng (VLM)", "KIE", "kie", True,
                                       "KIE đã có lưới hàng hóa — giữ KIE"))
    elif vlm_items:
        chosen_payload["items"] = [item.model_dump() for item in vlm_items]
        sources["items"] = "vlm"
        accepted.append("items")
        details_patch["items"] = {
            "value": len(vlm_items),
            "kie_conf": None,
            "ocr_conf": None,
            "method": "vlm_local",
            "count": len(vlm_items),
            "source": "vlm",
        }
    else:
        sources["items"] = "kie" if kie_items else "none"

    chosen_payload["field_confidence"] = dict(kie.field_confidence or {})
    for name in accepted:
        # VLM-sourced fields get NO fabricated confidence score — the number
        # would be fake precision. Field provenance (source + evidence) is the
        # reviewer's signal instead.
        chosen_payload["field_confidence"].pop(name, None)
    if kie_items:
        for index, item in enumerate(chosen_payload["items"]):
            chosen_payload["items"][index]["confidence"] = kie_items[index].confidence
            chosen_payload["items"][index]["bbox"] = kie_items[index].bbox

    data = InvoiceData.model_validate(chosen_payload)
    return ReconciliationResult(data, sources, conflicts, accepted, rejected), details_patch


def _conflict(
    name: str,
    kie_value: Any,
    vlm_value: Any,
    chosen: Any,
    chosen_source: str,
    resolved: bool,
    reason: str,
) -> dict[str, Any]:
    def render(value: Any) -> Any:
        return value.isoformat(timespec="seconds") if isinstance(value, datetime) else value

    return {
        "field": name,
        "kie_value": render(kie_value),
        "vlm_value": render(vlm_value),
        "chosen": render(chosen),
        "chosen_source": chosen_source,
        "resolved": resolved,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

class HybridPipeline:
    """Upload → preprocess → PaddleOCR → Spatial KIE → validation → quality
    gate → (local VLM fallback) → reconciliation → re-validation.

    Composition, not inheritance: the heuristic pipeline is reused untouched
    for steps 1–5 (which keeps OCR/KIE behavior identical to OCR+KIE mode),
    then a quality gate decides whether the VLM runs at all. When the gate
    passes, the reconciled draft replaces only the fields the rules allow and
    validation runs again on the merged invoice.
    """

    strategy = "hybrid-paddleocr-kie-vlm"

    def __init__(
        self,
        base: "HeuristicPipeline | None" = None,  # type: ignore[name-defined]
        vlm: "VLMExtractor | None" = None,  # type: ignore[name-defined]
        critical_fields: tuple[str, ...] | None = None,
        min_kie_confidence: float | None = None,
    ):
        from ..config import get_settings  # local import keeps module import cheap
        from .pipeline import HeuristicPipeline  # noqa: F401 — used in default
        from .vlm_extractor import VLMExtractor as _VLMExtractor

        settings = get_settings()
        self.base = base or HeuristicPipeline()
        self.vlm = vlm or _VLMExtractor(
            settings.vlm_provider,
            settings.openai_api_key or settings.gemini_api_key,
            settings.vlm_model,
        )
        raw_fields = critical_fields or tuple(
            item.strip() for item in settings.hybrid_critical_fields.split(",") if item.strip()
        )
        self.critical_fields = raw_fields or CRITICAL_FIELDS
        self.min_kie_confidence = (
            float(min_kie_confidence)
            if min_kie_confidence is not None
            else float(settings.hybrid_min_kie_confidence)
        )
        self._settings = settings

    def run(
        self,
        image_path,
        options=None,
        source_name: str | None = None,
    ):
        from pathlib import Path

        from .pipeline import _normalize_invoice
        from .postprocessor import validate_invoice

        settings = self._settings
        base_response = self.base.run(image_path, options=options, source_name=source_name)
        timings = dict(base_response.timings or {})
        warnings = list(base_response.warnings or [])
        details = {key: dict(value) for key, value in (base_response.field_details or {}).items()}
        invoice = base_response.invoice
        sources: dict[str, str] = {name: "kie" for name in details}
        conflicts: list[dict[str, Any]] = []
        fallback_reasons: list[str] = []
        vlm_triggered = False
        vlm_note: dict[str, Any] | None = None

        assessment = assess_quality(
            invoice,
            base_response.confidence,
            bool(base_response.math and base_response.math.passed),
            settings.review_confidence_threshold,
            details,
        )
        fallback_reasons = list(assessment.triggers)
        vlm_error: str | None = None

        if assessment.needs_vlm:
            available, reason = self._vlm_available()
            vlm_note = {
                "provider": self.vlm.provider,
                "model": (self._settings.vlm_local_model if self.vlm.provider == "local" else self.vlm.model),
            }
            if not available:
                warnings.append(
                    "Hybrid: cần VLM dự phòng nhưng chưa khả dụng"
                    + (f" ({reason})" if reason else "")
                    + " — giữ kết quả OCR+KIE."
                )
                vlm_note["skipped_reason"] = reason or "not configured"
            else:
                aligned_path = self._aligned_artifact(base_response.aligned_image_url)
                if aligned_path is None:
                    warnings.append("Hybrid: không tìm thấy ảnh đã căn chỉnh để chạy VLM — giữ OCR+KIE.")
                else:
                    vlm_triggered = True
                    import time as _time

                    step = _time.perf_counter()
                    try:
                        vlm_data, _vlm_boxes, vlm_raw = self.vlm.extract(aligned_path)
                        timings["vlm_ms"] = round((_time.perf_counter() - step) * 1000, 1)
                        evidence = self._extract_evidence(vlm_raw)
                        vlm_data = _normalize_invoice(vlm_data)
                        result, patch = reconcile(invoice, details, vlm_data, evidence)
                        invoice = result.data
                        conflicts = result.conflicts
                        sources.update(result.field_sources)
                        details.update(patch)
                        for name in result.vlm_accepted:
                            sources[name] = "vlm"
                        for item in result.vlm_rejected:
                            warnings.append(f"Hybrid: bỏ qua giá trị VLM cho “{item['field']}” — {item['reason']}.")
                    except Exception as exc:  # VLM failure must degrade, not crash
                        timings["vlm_ms"] = round((_time.perf_counter() - step) * 1000, 1)
                        vlm_triggered = False
                        vlm_error = str(exc)
                        warnings.append(f"Hybrid: VLM dự phòng lỗi ({exc}) — giữ kết quả OCR+KIE.")
                    vlm_note["triggered"] = vlm_triggered
                    vlm_note["error"] = vlm_error
                    vlm_note.update(self._vlm_stats())

        validation = validate_invoice(
            invoice, base_response.confidence, settings.review_confidence_threshold
        )
        status = validation.status
        warning_messages = list(warnings)
        for conflict in conflicts:
            if conflict.get("resolved"):
                warning_messages.append(
                    f"Hybrid · bất đồng (đã xử lý): {conflict['field']} — KIE “{conflict['kie_value']}” vs "
                    f"VLM “{conflict['vlm_value']}”; giữ {conflict['chosen_source'].upper()} theo quy tắc."
                )
            else:
                warning_messages.append(
                    f"Hybrid · xung đột cần soát: {conflict['field']} — KIE “{conflict['kie_value']}” vs "
                    f"VLM “{conflict['vlm_value']}”; tạm dùng {conflict['chosen_source'].upper()}."
                )
        if any(not conflict.get("resolved") for conflict in conflicts):
            status = "NEEDS_REVIEW"
        if vlm_triggered and assessment.triggers:
            warning_messages.append(
                "Hybrid: VLM dự phòng đã chạy vì " + "; ".join(assessment.describe().split("; ")) + "."
            )

        engine_info = dict(base_response.engine or {})
        engine_info["vlm"] = {
            **(vlm_note or {}),
            "strategy": "local-qwen" if self.vlm.provider == "local" else self.vlm.provider,
        }
        return base_response.model_copy(
            update={
                "invoice": invoice,
                "status": status,
                "warnings": warning_messages,
                "math": validation.math,
                "strategy": self.strategy,
                "timings": timings,
                "engine": engine_info,
                "field_details": details,
                "vlm_triggered": vlm_triggered,
                "fallback_reasons": fallback_reasons,
                "conflicts": conflicts,
                "sources": sources,
            }
        )

    # ------------------------------------------------------------------ helpers

    def _vlm_available(self) -> tuple[bool, str]:
        delegate = getattr(self.vlm, "fallback_available", None)
        if callable(delegate):
            return delegate()
        provider = (self.vlm.provider or "mock").lower()
        if provider in {"openai", "gemini", "google"}:
            return (True, "") if self.vlm.api_key else (False, "thiếu API key")
        return False, f"provider {provider!r} không hỗ trợ fallback"

    def _vlm_device(self) -> str | None:
        return self._vlm_stats().get("device")

    def _vlm_stats(self) -> dict[str, Any]:
        """Runtime stats of the local worker (device, VRAM peak, tokens...)."""
        if (self.vlm.provider or "").lower() != "local":
            return {}
        local = getattr(self.vlm, "_local", None)
        describe = getattr(local, "describe", None)
        if describe is None:
            return {}
        try:
            info = describe() or {}
        except Exception:
            return {}
        return {
            key: info.get(key)
            for key in ("device", "last_vram_peak_mb", "last_visual_tokens", "last_ms", "loaded")
            if info.get(key) is not None
        }
        return False, f"provider {provider!r} không hỗ trợ fallback"

    def _aligned_artifact(self, url: str | None):
        from pathlib import Path

        if not url:
            return None
        path = Path(self._settings.processed_dir) / Path(url).name
        return path if path.is_file() else None

    @staticmethod
    def _extract_evidence(raw_text: str) -> dict[str, str]:
        import json as _json

        try:
            payload = _json.loads(raw_text)
        except (TypeError, ValueError):
            return {}
        evidence = payload.get("evidence") if isinstance(payload, dict) else None
        if not isinstance(evidence, dict):
            return {}
        return {str(key): str(value) for key, value in evidence.items() if value}

