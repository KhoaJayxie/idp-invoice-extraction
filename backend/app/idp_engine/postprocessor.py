from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata
from typing import Any

from ..schemas.invoice import InvoiceData, MathCheck


# Currency markers by script family. Dot-decimal currencies ("RM", "$", "USD")
# read "9.000" as nine units; Vietnamese dong reads "9.000" as nine thousand.
DOT_DECIMAL_CURRENCIES = ("rm", "usd", "sgd", "us$", "$", "€", "£")
DOT_THOUSAND_CURRENCIES = ("vnd", "vnđ", "₫", "đ")


def currency_hint(text: str) -> str | None:
    """Return a currency code detected in `text` ('VND', 'MYR', 'USD', ...).

    Only markers adjacent to digits/word boundaries count: 'đ' inside a
    Vietnamese word ('Đường') is a letter, not a dong symbol.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if "₫" in folded or re.search(r"\d\s*đ(?![a-zà-ỹ])", folded) or re.search(r"(?<![a-zà-ỹ])đ\s*\d", folded) or re.search(r"(?<![a-z0-9])vnd(?![a-z0-9])", folded):
        return "VND"
    if re.search(r"(?<![a-z])(?:rm|myr)(?![a-z])", folded):
        return "MYR"
    if "$" in folded or re.search(r"(?<![a-z0-9])usd(?![a-z0-9])", folded):
        return "USD"
    if "€" in folded:
        return "EUR"
    if "£" in folded:
        return "GBP"
    if re.search(r"(?<![a-z0-9])sgd(?![a-z0-9])", folded):
        return "SGD"
    return None


CURRENCY_CODES = {"rm": "MYR", "$": "USD", "usd": "USD", "€": "EUR", "£": "GBP", "sgd": "SGD", "us$": "USD"}


@dataclass(slots=True)
class ValidationResult:
    status: str
    confidence: float
    warnings: list[str]
    math: MathCheck


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace while retaining Vietnamese accents."""
    value = unicodedata.normalize("NFC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_money(value: Any, decimal_dot: bool | None = None) -> float:
    """Parse Vietnamese/international money strings into a numeric amount.

    Vietnamese receipts commonly use dot as a thousands separator (93.500),
    while dot-decimal currencies (RM, USD) use dot as the decimal mark
    ("9.000" = nine). OCR O/o mistakes in numeric-looking values are
    corrected. `decimal_dot` forces one reading when the caller knows the
    currency context; None applies mixed heuristics.
    """
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raw = unicodedata.normalize("NFKC", str(value)).strip().upper()
    if not raw:
        return 0.0
    raw = raw.replace("O", "0")
    negative = raw.startswith("-") or (raw.startswith("(") and raw.endswith(")"))
    raw = re.sub(r"[^0-9,.]", "", raw)
    if not raw:
        return 0.0
    if "," in raw and "." in raw:
        # Rightmost punctuation is interpreted as decimal; preceding grouping
        # punctuation is removed. Vietnamese and English forms both work.
        decimal_mark = "," if raw.rfind(",") > raw.rfind(".") else "."
        grouping_mark = "." if decimal_mark == "," else ","
        raw = raw.replace(grouping_mark, "").replace(decimal_mark, ".")
    elif "," in raw:
        parts = raw.split(",")
        if len(parts) > 2:
            # Repeated punctuation followed by exactly 1-2 digits is decimal;
            # otherwise all punctuation is a grouping separator.
            raw = "".join(parts[:-1]) + ("." + parts[-1] if len(parts[-1]) in (1, 2) else parts[-1])
        elif len(parts[-1]) == 3:
            raw = "".join(parts)
        else:
            raw = ".".join(parts)
    elif "." in raw:
        parts = raw.split(".")
        if len(parts) > 2:
            raw = "".join(parts[:-1]) + ("." + parts[-1] if len(parts[-1]) in (1, 2) else parts[-1])
        elif decimal_dot is True:
            pass  # dot-decimal currency: "9.000" RM stays nine
        elif decimal_dot is False or len(parts[-1]) == 3:
            # Vietnamese thousands reading wins when forced or ambiguous.
            raw = "".join(parts)
    try:
        amount = float(raw)
    except ValueError:
        return 0.0
    return -amount if negative else amount


def normalize_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = normalize_text(value)
    if not text:
        return None
    text = text.replace("T", " ").replace("年", "-").replace("月", "-").replace("日", " ")
    # 12-hour clocks: "8:13:39 PM", "08:32 AM" — fold into 24-hour first so
    # every downstream pattern stays uniform.
    meridiem = re.search(r"\b(\d{1,2})[:h](\d{2})(?::(\d{2}))?\s*([AP])\.?\s*M?\b", text, flags=re.IGNORECASE)
    if meridiem:
        hour, minute, second, ampm = meridiem.groups()
        hour = int(hour) % 12 + (12 if ampm.upper() == "P" else 0)
        replacement = f"{hour:02d}:{minute}" + (f":{second}" if second else "")
        text = text[: meridiem.start()] + replacement + text[meridiem.end() :]
    candidates = (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H%M",
        "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%d-%m-%Y", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%d/%m/%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    )
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    match = re.search(
        r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](20\d{2})(?:\s+(\d{1,2})[:h](\d{2})(?:[:](\d{2}))?)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        day, month, year, hour, minute, second = match.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0))
        except ValueError:
            return None
    # Vietnamese long form: "Ngày 16 tháng 6 năm 2025" (optionally with time).
    # OCR may drop the diacritics ("ngay 16 thang 6 nam 2025").
    match = re.search(
        r"ng[àa]y\s+(\d{1,2})\s+th[áa]ng\s+(\d{1,2})\s+n[ăa]m\s+(20\d{2})(?:\s+(\d{1,2})[:h](\d{2}))?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        day, month, year, hour, minute = match.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))
        except ValueError:
            return None
    match = re.search(
        r"(?<!\d)(20\d{2})[/.\-](\d{1,2})[/.\-](\d{1,2})(?:\s+(\d{1,2})[:h](\d{2}))?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        year, month, day, hour, minute = match.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))
        except ValueError:
            return None
    return None


def validate_invoice(
    data: InvoiceData,
    confidence: float,
    threshold: float = 0.85,
    tolerance_ratio: float = 0.01,
) -> ValidationResult:
    warnings: list[str] = []
    # Individual item amounts are the ground truth of the goods grid; the
    # reported "tổng phụ"/subtotal block is frequently misread by OCR (a
    # single dropped/merged column corrupts it). Derived subtotals also keep
    # the math check meaningful for receipts without a subtotal line.
    items_sum = round(sum(item.amount if item.amount else item.quantity * item.unit_price for item in data.items), 2)
    line_sum = items_sum if data.items else float(data.subtotal)
    subtotal = items_sum if data.items else float(data.subtotal)
    if items_sum and data.subtotal and abs(items_sum - data.subtotal) > max(1.0, items_sum * 0.01):
        warnings.append(
            f"Tạm tính ghi trên hóa đơn ({data.subtotal:.0f}) lệch với tổng dòng hàng ({items_sum:.0f}); dùng tổng dòng hàng."
        )
    expected = round(subtotal + data.tax - data.discount, 2)
    # Cash rounding ("Rounding (RM): 0.01", "Rounding Adjustment: 0.00") moves
    # the printed total away from the exact arithmetic sum; accept a small
    # rounding gap on top of the percentage tolerance.
    rounding_gap = min(abs(float(data.total_cost)), 0.05) if data.currency and data.currency != "VND" else 0.0
    tolerance = max(1.0, abs(data.total_cost) * tolerance_ratio) + rounding_gap
    passed = data.total_cost > 0 and abs(expected - data.total_cost) <= tolerance
    failed_check = data.total_cost > 0 and (data.items or data.subtotal > 0) and not passed
    if failed_check:
        warnings.append(
            f"Tổng không khớp: hàng hóa + thuế - giảm giá = {expected:.2f}, hóa đơn ghi {data.total_cost:.2f}."
        )
    elif data.total_cost <= 0:
        warnings.append("Chưa rút trích được tổng thanh toán.")
    if confidence < threshold:
        warnings.append(f"Độ tin cậy {confidence:.2f} thấp hơn ngưỡng {threshold:.2f}.")
    if not data.seller:
        warnings.append("Chưa xác định được tên đơn vị bán.")
    if not data.items:
        warnings.append("Chưa phát hiện được dòng hàng hóa.")
    math = MathCheck(
        line_sum=line_sum,
        tax=float(data.tax),
        discount=float(data.discount),
        expected_total=expected,
        actual_total=float(data.total_cost),
        delta=round(float(data.total_cost) - expected, 2),
        passed=passed,
        tolerance=round(tolerance, 2),
    )
    return ValidationResult("NEEDS_REVIEW" if warnings else "VALIDATED", float(confidence), warnings, math)
