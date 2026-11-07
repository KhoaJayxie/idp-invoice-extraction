from __future__ import annotations

from datetime import datetime

import pytest

from backend.app.idp_engine.kie_extractor import KIEExtractor
from backend.app.idp_engine.ocr_detector import OCRDetector, OCRToken, OCRUnavailableError
from backend.app.idp_engine.postprocessor import normalize_datetime, normalize_money, validate_invoice
from backend.app.schemas.invoice import InvoiceData, LineItem


def test_money_and_date_normalization():
    assert normalize_money("93.500 ₫") == 93500
    assert normalize_money("1.234,56") == 1234.56
    assert normalize_money("1O.OOO") == 10000
    assert normalize_datetime("22/10/2026 14:30") == datetime(2026, 10, 22, 14, 30)


def test_vietnamese_long_form_date_parses():
    assert normalize_datetime("Ngày 16 tháng 6 năm 2025") == datetime(2025, 6, 16)
    assert normalize_datetime("ngay 16 thang 6 nam 2025") == datetime(2025, 6, 16)


def test_math_validation_marks_mismatch_and_low_confidence():
    data = InvoiceData(
        seller="ABC",
        tax=8500,
        items=[LineItem(name="Cà phê", quantity=2, unit_price=30000, amount=60000)],
        total_cost=100000,
    )
    result = validate_invoice(data, 0.9)
    assert result.status == "NEEDS_REVIEW"
    assert result.math.expected_total == 68500
    assert result.math.passed is False
    assert result.warnings


def test_validation_prefers_item_sum_over_misread_subtotal():
    data = InvoiceData(
        seller="ABC",
        subtotal=3000678.0,  # OCR mangled the subtotal block
        items=[
            LineItem(name="A", quantity=1, unit_price=3000000, amount=3000000),
            LineItem(name="B", quantity=2, unit_price=3000000, amount=6000000),
            LineItem(name="C", quantity=1, unit_price=3000000, amount=3000000),
        ],
        total_cost=12000000,
    )
    result = validate_invoice(data, 0.97)
    assert result.math.expected_total == 12000000
    assert result.math.passed is True
    assert any("Tạm tính" in warning for warning in result.warnings)


def test_kie_extracts_vietnamese_header_and_items():
    tokens = [
        OCRToken("CỬA HÀNG MINH AN", [0, 0, 600, 30], 0.98),
        OCRToken("Địa chỉ: 123 Nguyễn Huệ, TP.HCM", [0, 40, 600, 70], 0.95),
        OCRToken("Số hóa đơn: HD-001", [0, 80, 600, 110], 0.95),
        OCRToken("Ngày: 22/10/2026 14:30", [0, 120, 600, 150], 0.95),
        OCRToken("Cà phê sữa 2 30.000 60.000", [0, 200, 700, 230], 0.92),
        OCRToken("Thuế VAT: 8.500", [0, 250, 600, 280], 0.92),
        OCRToken("Tổng cộng: 68.500", [0, 300, 600, 330], 0.92),
    ]
    data, boxes, raw_text = KIEExtractor().extract(tokens)
    assert data.seller == "CỬA HÀNG MINH AN"
    assert data.address == "123 Nguyễn Huệ, TP.HCM"
    assert data.invoice_id == "HD-001"
    assert data.timestamp == datetime(2026, 10, 22, 14, 30)
    assert data.tax == 8500
    assert data.total_cost == 68500
    assert len(data.items) == 1
    assert data.items[0].amount == 60000
    assert len(boxes) == len(tokens)
    assert "Cà phê sữa" in raw_text


def test_kie_merges_column_tokens_into_visual_lines():
    """PP-OCR det returns one box per column cell; KIE must reassemble rows."""
    tokens = [
        OCRToken("Tng cng:", [620, 1046, 800, 1090], 1.0),      # OCR dropped vowels
        OCRToken("12.000.000đ", [862, 1052, 1050, 1100], 0.965),
        OCRToken("Tip cn (Tiu thuyt)", [126, 660, 400, 700], 0.997),
        OCRToken("1", [421, 668, 435, 700], 1.0),
        OCRToken("3.000.000đ", [624, 668, 760, 700], 0.966),
        OCRToken("3.000.000đ", [860, 668, 1000, 700], 0.948),
    ]
    data, boxes, raw_text = KIEExtractor().extract(tokens)
    # Column cells merged into one line before the rules run.
    assert "Tip cn (Tiu thuyt) 1 3.000.000đ 3.000.000đ" in raw_text
    assert len(data.items) == 1
    assert data.items[0].quantity == 1
    assert data.items[0].unit_price == 3000000
    assert data.items[0].amount == 3000000
    # Fuzzy label recovery for the vowel-dropped total label.
    assert data.total_cost == 12000000
    # Overlay still receives one box per original token.
    assert len(boxes) == len(tokens)


def test_kie_reads_glued_currency_and_ocr_d_suffix():
    tokens = [
        OCRToken("Chic áo màu đen 2 3.000.000đ 6.000.000đ", [100, 700, 950, 760], 0.94),
        OCRToken("Kẹo dừa 1 3.000.000d 3.000.000d", [100, 780, 950, 830], 0.9),
    ]
    data, _, _ = KIEExtractor().extract(tokens)
    assert [item.amount for item in data.items] == [6000000, 3000000]
    assert data.items[1].unit_price == 3000000


def test_kie_rejects_phone_numbers_as_items():
    tokens = [
        OCRToken("123 Đưng ABC, +84 912 345 678", [100, 1400, 700, 1450], 0.95),
        OCRToken("S tài khon: 0962.999.772", [100, 1460, 700, 1500], 0.95),
    ]
    data, _, _ = KIEExtractor().extract(tokens)
    assert data.items == []


def test_kie_three_field_rows_and_wrapped_parenthetical_names():
    tokens = [
        OCRToken("(Tiu thuyt tranh nh) Chic áo màu đen 2 6.000.000đ", [100, 700, 950, 800], 0.95),
        OCRToken("Cà phê 2 60.000", [100, 820, 500, 860], 0.95),
    ]
    data, _, _ = KIEExtractor().extract(tokens)
    by_name = {item.name: item for item in data.items}
    assert by_name["Chic áo màu đen"].quantity == 2
    assert by_name["Chic áo màu đen"].unit_price == 3000000
    assert by_name["Cà phê"].amount == 60000


def test_sidecar_boxes_spread_across_full_height():
    from pathlib import Path

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "receipt.png"
        image_path.write_bytes(b"fake")
        image_path.with_suffix(".txt").write_text(
            "DÒNG MỘT\nDÒNG HAI\nDÒNG BA\nDÒNG BỐN\n", encoding="utf-8"
        )
        height, width = 1600, 1131
        tokens = OCRDetector._read_sidecar(image_path, width=width, height=height)
        assert len(tokens) == 4
        assert tokens[0].box[1] == 0
        assert tokens[-1].box[1] == pytest.approx(height * 3 / 4, rel=0.01)
        assert tokens[-1].box[3] <= height
        # Boxes span the full row band instead of stacking in the top 28px.
        assert all(token.box[3] - token.box[1] > height / 8 for token in tokens)


def test_strict_real_mode_raises_when_paddle_missing():
    detector = OCRDetector.__new__(OCRDetector)
    detector.use_paddle = True
    detector.ocr = None
    detector.vietocr_predictor = None
    detector.backend_warning = "PaddleOCR chưa sẵn sàng: thiếu wheel"
    detector.strict = True
    with pytest.raises(OCRUnavailableError):
        detector.detect([])


def test_non_strict_detector_falls_back_to_sidecar(tmp_path):
    import numpy as np
    from backend.app.idp_engine.preprocessor import PreprocessResult, PreprocessMetadata

    detector = OCRDetector(use_paddle=False, strict=False)
    image_path = tmp_path / "receipt.png"  # noqa: F841 - sidecar is looked up from this path
    image_path.with_suffix(".txt").write_text("Tổng cộng: 60.000\n", encoding="utf-8")
    gray = np.full((200, 300), 255, dtype=np.uint8)
    processed = PreprocessResult(
        original_bgr=np.full((200, 300, 3), 255, dtype=np.uint8),
        enhanced_gray=gray,
        binary=np.zeros_like(gray),
        metadata=PreprocessMetadata(
            original_width=300, original_height=200, processed_width=300, processed_height=200
        ),
    )
    tokens = detector.detect(processed, source_path=image_path)
    assert len(tokens) == 1
    assert tokens[0].source == "sidecar"
