"""Unit tests for the spatial KIE engine.

Two kinds of evidence:

* synthetic tokens laid out on a simple grid (unit coverage of spatial
  grouping, column-role assignment, disambiguation, null-on-insufficient);
* the real PaddleOCR token dumps under ``evaluation/fixtures/ocr_tokens``
  (no mocks) — regression coverage for the demo invoices the extractor was
  tuned against.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backend.app.idp_engine.kie_extractor import KIEExtractor
from backend.app.idp_engine.ocr_detector import OCRToken
from backend.app.idp_engine.postprocessor import normalize_datetime, normalize_money

FIXTURES = Path(__file__).resolve().parents[2] / "evaluation" / "fixtures" / "ocr_tokens"
CHAR_W = 8.0
LINE_H = 24.0


def tok(text: str, x: float, y: float, w: float | None = None, conf: float = 0.97) -> OCRToken:
    width = w if w is not None else max(18.0, len(text) * CHAR_W)
    return OCRToken(text, [x, y, x + width, y + LINE_H * 0.8], conf)


def row_text(y: float, text: str, x: float = 60.0) -> list[OCRToken]:
    """Lay words left-to-right starting at x; gaps stay same-line."""
    tokens: list[OCRToken] = []
    cursor = x
    for word in text.split():
        w = max(18.0, len(word) * CHAR_W)
        tokens.append(tok(word, cursor, y))
        cursor += w + 12.0
    return tokens


def right_col(y: float, text: str, x: float = 620.0) -> list[OCRToken]:
    """A far-right column segment on the same visual row (gap >= 110px)."""
    return row_text(y, text, x)


# Column anchors for the synthetic goods grid (page ~800px wide).
COLS = [60.0, 320.0, 520.0, 700.0]  # desc / qty / price / amount


def table_row(y: float, cells: list[str], xs: list[float] = COLS) -> list[OCRToken]:
    """One goods-grid row: each non-empty cell at its column anchor."""
    tokens: list[OCRToken] = []
    for cell, x in zip(cells, xs):
        if cell:
            w = max(18.0, len(cell) * CHAR_W)
            tokens.append(tok(cell, x, y, w))
    return tokens


def load_fixture(name: str) -> list[OCRToken]:
    dump = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return [OCRToken(t["text"], t["box"], t["confidence"]) for t in dump]


def extract(tokens: list[OCRToken]):
    data, boxes, raw, details = KIEExtractor().extract_with_details(tokens)
    return data, details


# --------------------------------------------------------------- spatial

def test_same_row_tokens_group_into_one_line_and_columns_split() -> None:
    tokens = row_text(100.0, "BILL TO: ACME SDN BHD") + right_col(100.0, "Receipt No: TD01167104")
    data, _ = extract(tokens)
    assert data.invoice_id == "TD01167104"
    # The right segment was a label, not the buyer value.
    assert data.buyer is None or "RECEIPT" not in data.buyer.upper()


def test_label_below_value_same_column() -> None:
    tokens = (
        row_text(100.0, "HÓA ĐƠN ĐƯỢC GỬI CHO:")
        + row_text(130.0, "Công ty An Phú")
        + row_text(160.0, "123 Đưng ABC, Thành phố DEF")
        + row_text(190.0, "S điện thoại: +84 912 345 678")
    )
    data, _ = extract(tokens)
    assert data.buyer == "Công ty An Phú"
    assert data.buyer_address == "123 Đưng ABC, Thành phố DEF"


def test_fuzzy_label_tolerates_ocr_noise() -> None:
    tokens = row_text(100.0, "Recelpt #: CS00082258")
    data, _ = extract(tokens)
    assert data.invoice_id == "CS00082258"


def test_null_when_no_evidence() -> None:
    tokens = row_text(100.0, "Xin chào quý khách") + row_text(130.0, "Cảm ơn quý khách")
    data, details = extract(tokens)
    assert data.seller is None
    assert data.invoice_id is None
    assert data.total_cost == 0.0
    assert data.items == []
    assert details == {}


# --------------------------------------------------------------- table

TABLE_HEAD = "Item Description Qty Unit Price Amount"


def test_table_columns_and_math() -> None:
    tokens = (
        table_row(100.0, ["Item Description", "Qty", "Unit Price", "Amount"])
        + table_row(130.0, ["Coffee", "2", "30.000", "60.000"])
        + table_row(160.0, ["Milk", "1", "50.000", "50.000"])
        + row_text(190.0, "Total: 110.000")
    )
    data, _ = extract(tokens)
    assert len(data.items) == 2
    assert data.items[0].quantity == 2.0
    assert data.items[0].amount == 60000.0
    assert data.items[1].quantity == 1.0


def test_wrapped_price_continuation_joins_item() -> None:
    tokens = (
        table_row(100.0, ["Item Description", "Qty", "Unit Price", "Amount"])
        + table_row(130.0, ["Sách giáo khoa", "2", "", "6.000.000đ"])
        + table_row(145.0, ["", "", "3.000.000đ", ""])
        + row_text(190.0, "Tng cng: 6.000.000đ")
    )
    data, _ = extract(tokens)
    assert len(data.items) == 1
    item = data.items[0]
    assert item.quantity == 2.0
    assert item.unit_price == 3000000.0
    assert item.amount == 6000000.0


def test_desc_column_bare_integer_is_quantity_not_code() -> None:
    tokens = (
        table_row(100.0, ["Item Description", "Qty", "Unit Price", "Amount"])
        + table_row(130.0, ["Tip cn (Tiu thuyt) 1", "", "3.000.000đ", "3.000.000đ"])
        + row_text(190.0, "Tng cng: 3.000.000đ")
    )
    data, _ = extract(tokens)
    assert len(data.items) == 1
    assert data.items[0].quantity == 1.0
    assert "1" not in data.items[0].name


# --------------------------------------------------------------- identifiers

def test_tax_id_patterns() -> None:
    for value in ("789417-W", "000800689824", "0311234567", "1234567890123"):
        tokens = row_text(100.0, f"GST No: {value}")
        data, _ = extract(tokens)
        assert data.tax_id == value, value


def test_phone_is_not_a_tax_id() -> None:
    tokens = row_text(100.0, "Điện thoại: +84 912 345 678")
    data, _ = extract(tokens)
    assert data.tax_id is None


# --------------------------------------------------------------- dates / money

def test_ampm_folds_into_24h() -> None:
    parsed = normalize_datetime("2018-12-25 8:13:39 PM")
    assert parsed == datetime(2018, 12, 25, 20, 13, 39)


def test_same_segment_time_fold() -> None:
    """One OCR block holds 'Date. 09/02/2018 Time:08:32:00' (zhenghui demo)."""
    tokens = row_text(100.0, "Date. 09/02/2018 Time:08:32:00")
    data, _ = extract(tokens)
    assert data.timestamp == datetime(2018, 2, 9, 8, 32, 0)


def test_dot_decimal_currencies_vs_vnd() -> None:
    assert normalize_money("9.000", decimal_dot=True) == 9.0      # RM 9.000
    assert normalize_money("3.000.000", decimal_dot=False) == 3000000.0  # VND
    # End to end: the KIE reads the currency from the line ("RM") before
    # parsing, so a Malay receipt total keeps its dot-decimal reading.
    tokens = row_text(100.0, "BOOK SHOP SDN BHD") + right_col(100.0, "Total RM 9.000")
    data, _ = extract(tokens)
    assert data.total_cost == 9.0
    assert data.currency == "MYR"


def test_total_avoids_gst_line() -> None:
    tokens = (
        row_text(130.0, "Subtotal RM 411.50")
        + row_text(160.0, "GST (6%) RM 24.69")
        + row_text(190.0, "Rounding RM 0.01")
        + row_text(220.0, "Total RM 436.20")
    )
    data, _ = extract(tokens)
    assert data.total_cost == 436.20
    assert data.tax == 24.69


# --------------------------------------------------------------- details

def test_field_details_shape() -> None:
    data, details = extract(load_fixture("sroie_bookta"))
    assert set(details) >= {"seller", "invoice_id", "timestamp", "total_cost", "currency"}
    for name, detail in details.items():
        if name == "items":
            continue  # aggregate entry carries a count, not box ids
        assert 0.0 < detail["kie_conf"] <= 1.0, name
        assert 0.0 <= detail["ocr_conf"] <= 1.0, name
        assert detail["method"], name
        assert isinstance(detail["box_ids"], list), name
    assert data.field_confidence["invoice_id"] == details["invoice_id"]["kie_conf"]


# --------------------------------------------------------------- real-OCR regression

def test_regression_bookta_receipt() -> None:
    """USER DEMO regression: BOOK TA receipt (data/uploads/...X00016469612).

    Before the fix: seller='tan woon yann', invoice_id=None, items=0, total=9.0.
    """
    data, details = extract(load_fixture("sroie_bookta"))
    assert data.seller == "BOOK TA K (TAMAN DAYA) SDN BHD"
    assert data.invoice_id == "TD01167104"
    assert data.tax_id == "789417-W"
    assert data.address == "NO.5: 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU"
    assert data.timestamp == datetime(2018, 12, 25, 20, 13, 39)
    assert data.total_cost == 9.0
    assert data.currency == "MYR"
    assert len(data.items) == 1
    assert data.items[0].quantity == 1.0
    assert data.items[0].amount == 9.0


def test_regression_zhenghui_invoice() -> None:
    """USER DEMO regression: ZHENG HUI invoice (data/uploads/...X51005200931).

    Before: seller='h02038', subtotal=411.5, tax=0, total=411.5, items=0.
    """
    data, _ = extract(load_fixture("sroie_zhenghui"))
    assert data.seller == "PERNIAGAAN ZHENG HUI"
    assert data.buyer == "GOGIANT ENGINEERING (M) SDN BHD"
    assert data.tax_id == "000800689824"
    assert data.invoice_id == "CS00082258"
    assert data.address == "NO.59 JALAN PERMAS 9/5, BANDAR BARU PERMAS JAYA, 81750 JOHOR BAHRU"
    assert data.buyer_address is None
    assert data.timestamp == datetime(2018, 2, 9, 8, 32, 0)
    assert data.subtotal == 411.5
    assert data.tax == 24.69
    assert data.total_cost == 436.2
    assert data.currency == "MYR"
    assert len(data.items) == 6
    assert data.items[0].quantity == 5.0
    assert data.items[0].amount == 17.5


def test_regression_vn_bookstore() -> None:
    """VN bookstore receipt: two-column header + wrapped product rows."""
    data, _ = extract(load_fixture("vn_bookstore_1"))
    assert data.seller == "NHÀ XUÃT BÁN HUYÊN THOAI"
    assert data.buyer == "Công ty An Phú"
    assert data.buyer_address == "123 Đưng ABC, Thành ph DEF"
    assert data.invoice_id is None  # the receipt carries no invoice number
    assert data.timestamp == datetime(2025, 6, 16)
    assert data.total_cost == 12000000.0
    assert data.currency == "VND"
    assert len(data.items) == 3
    wrapped = data.items[1]
    assert wrapped.quantity == 2.0
    assert wrapped.unit_price == 3000000.0
    assert wrapped.amount == 6000000.0
