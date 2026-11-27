from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_png() -> bytes:
    image = np.full((240, 180, 3), 255, dtype=np.uint8)
    cv2.putText(image, "receipt", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_and_preprocess_preview(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    response = client.post(
        "/api/preprocess/preview",
        files={"file": ("receipt.png", make_png(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["original"].startswith("data:image/")
    assert body["metadata"]["processed_width"] > 0


def test_review_workflow_preserves_prediction_and_records_history(client):
    payload = {
        "invoice_id": "AI-001",
        "seller": "AI SHOP",
        "total_cost": 100.0,
        "status": "NEEDS_REVIEW",
        "confidence": 0.62,
        "items": [{"name": "Coffee", "quantity": 1, "unit_price": 100, "amount": 100, "confidence": 0.8}],
        "raw_json": {"boxes": [{"text": "AI SHOP", "box": [1, 2, 30, 10]}], "field_details": {"seller": {"box_ids": [0], "kie_conf": 0.7}}},
    }
    created = client.post("/api/invoices", json=payload)
    assert created.status_code == 201, created.text
    invoice_id = created.json()["id"]
    original = created.json()["original_prediction"]
    assert original["data"]["seller"] == "AI SHOP"

    updated = client.put(f"/api/invoices/{invoice_id}", json={"seller": "Human corrected shop"})
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["seller"] == "Human corrected shop"
    assert body["original_prediction"] == original
    assert any(event["type"] == "correction" for event in body["correction_history"])

    reviewed = client.post(f"/api/invoices/{invoice_id}/review", json={"status": "VERIFIED", "note": "Đã đối chiếu bản gốc"})
    assert reviewed.status_code == 200, reviewed.text
    body = reviewed.json()
    assert body["status"] == "VERIFIED"
    assert body["reviewed_at"]
    assert body["correction_history"][-1]["status"] == "VERIFIED"
    assert client.get(f"/api/invoices/{invoice_id}").json()["status"] == "VERIFIED"
    assert client.get(f"/api/invoices/{invoice_id}/review-history").json() == body["correction_history"]


def test_review_action_rejects_invalid_status(client):
    created = client.post("/api/invoices", json={"seller": "Shop", "status": "NEEDS_REVIEW"})
    assert created.status_code == 201
    response = client.post(f"/api/invoices/{created.json()['id']}/review", json={"status": "VALIDATED"})
    assert response.status_code == 422


def test_validation_blocks_verification_and_duplicate_override(client):
    payload = {
        "invoice_id": "P6-001",
        "seller": "P6 SHOP",
        "timestamp": "2026-01-01T10:00:00",
        "subtotal": 100,
        "tax": 8,
        "total_cost": 108,
        "confidence": 0.95,
        "file_hash": "a" * 64,
        "items": [{"name": "A", "quantity": 1, "unit_price": 100, "amount": 100, "confidence": 0.9}],
    }
    created = client.post("/api/invoices", json=payload)
    assert created.status_code == 201, created.text
    invoice_id = created.json()["id"]
    duplicate = client.post("/api/invoices", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "DUPLICATE_INVOICE"

    bad = client.put(f"/api/invoices/{invoice_id}", json={"total_cost": 999})
    assert bad.status_code == 200
    assert any(issue["code"] == "total_mismatch" for issue in bad.json()["validation_issues"])
    blocked = client.post(f"/api/invoices/{invoice_id}/review", json={"status": "VERIFIED"})
    assert blocked.status_code == 422
    verified = client.post(f"/api/invoices/{invoice_id}/review", json={"status": "VERIFIED", "validation_override": True, "validation_override_reason": "Đã đối chiếu bản giấy"})
    assert verified.status_code == 200, verified.text
    assert verified.json()["status"] == "VERIFIED"
    assert verified.json()["validation_override"] is True


def test_duplicate_override_requires_reason(client):
    base = {"invoice_id": "P6-002", "seller": "P6 SHOP", "file_hash": "b" * 64, "total_cost": 10, "validation_override": True, "validation_override_reason": "Đối chiếu thủ công"}
    assert client.post("/api/invoices", json=base).status_code == 201
    duplicate = client.post("/api/invoices", json={**base, "duplicate_override": True})
    assert duplicate.status_code == 422


def test_extract_returns_503_in_strict_real_mode_without_engine(client, monkeypatch):
    from backend.app.api import routes

    class FailingPipeline:
        strategy = "2-stage-paddleocr-vietocr-kie"

        def run(self, *args, **kwargs):
            raise routes.OCRUnavailableError("PaddleOCR chưa sẵn sàng: simulated")

    monkeypatch.setattr(routes, "_get_pipeline", lambda mode: FailingPipeline())
    response = client.post(
        "/api/extract",
        files={"file": ("receipt.png", make_png(), "image/png")},
        data={"ocr_text": "Tổng cộng: 60.000"},
    )
    assert response.status_code == 503
    assert "real mode" in response.json()["detail"]


def test_extract_boxes_stay_within_processed_bounds(client):
    text = "\n".join([
        "CỬA HÀNG ABC",
        "Số hóa đơn: HD-BOX-01",
        "Cà phê 2 30.000 60.000",
        "Tổng cộng: 60.000",
    ])
    response = client.post(
        "/api/extract",
        files={"file": ("receipt.png", make_png(), "image/png")},
        data={"ocr_text": text},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    width = body["preprocess"]["processed_width"]
    height = body["preprocess"]["processed_height"]
    assert body["boxes"]
    for box in body["boxes"]:
        x1, y1, x2, y2 = box["box"]
        assert 0 <= x1 <= width and 0 <= x2 <= width
        assert 0 <= y1 <= height and 0 <= y2 <= height
        assert x2 > x1 and y2 > y1

    text = "\n".join([
        "CỬA HÀNG ABC",
        "Địa chỉ: 1 Lê Lợi, Đà Nẵng",
        "Số hóa đơn: HD-API-01",
        "Ngày: 22/10/2026 14:30",
        "Cà phê 2 30.000 60.000",
        "Thuế VAT: 6.000",
        "Tổng cộng: 66.000",
    ])
    response = client.post(
        "/api/extract",
        files={"file": ("receipt.png", make_png(), "image/png")},
        data={"ocr_text": text},
    )
    assert response.status_code == 200, response.text
    extracted = response.json()
    assert extracted["invoice"]["invoice_id"] == "HD-API-01"
    assert extracted["invoice"]["total_cost"] == 66000
    assert extracted["status"] == "NEEDS_REVIEW"  # sidecar confidence is intentionally 0.55

    payload = extracted["invoice"] | {
        "status": "VALIDATED",
        "confidence": 0.99,
        "strategy": extracted["strategy"],
        "source_image": extracted["source_image"],
        "original_filename": extracted["original_filename"],
        "raw_json": {"raw_text": extracted["raw_text"]},
        "preprocessing_json": extracted["preprocess"],
    }
    created = client.post("/api/invoices", json=payload)
    assert created.status_code == 201, created.text
    invoice_id = created.json()["id"]
    assert client.get(f"/api/invoices/{invoice_id}").status_code == 200
    stats = client.get("/api/stats?days=all")
    assert stats.status_code == 200
    assert stats.json()["invoice_count"] >= 1
    assert client.get("/api/export?format=csv").status_code == 200
    assert client.get("/api/export?format=xlsx").status_code in {200, 503}
    assert client.delete(f"/api/invoices/{invoice_id}").status_code == 204
