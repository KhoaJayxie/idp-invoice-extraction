from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata
from typing import Any

from ..schemas.invoice import InvoiceData, MathCheck, ValidationIssue


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
    issues: list[ValidationIssue]

    @property
    def critical_errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "critical"]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity in {"error", "critical"}]


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
    # "17/01/2023 - 17:36": a dash between date and clock breaks the optional
    # time group of every numeric pattern — fold it into a plain space.
    text = re.sub(r"(\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4})\s*[-–]\s*(\d{1,2}:\d{2})", r"\1 \2", text)
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
    # OCR may drop the diacritics ("ngay 16 thang 6 nam 2025") and VAT
    # invoices interleave English labels in parentheses ("Ngày (date) 26
    # tháng (month) 06 năm (year) 2023").
    match = re.search(
        r"ng[àa]y\s*(?:\([^)]*\))?\s*(\d{1,2})\s*th[áàa]ng\s*(?:\([^)]*\))?\s*(\d{1,2})\s*n[ăaà]m\s*(?:\([^)]*\))?\s*(20\d{2})(?:\s+(?:[-–]\s+)?(\d{1,2})[:h](\d{2}))?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        day, month, year, hour, minute = match.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))
        except ValueError:
            return None
    # Short Vietnamese month form: "07 thg 5, 2023" / "7 thg 10 2017".
    match = re.search(
        r"(?<!\d)(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(20\d{2})(?:\s+(\d{1,2})[:h](\d{2}))?",
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
    """Validate invoice arithmetic and identity fields without treating zero as missing.

    Presence is taken from ``field_presence`` when extraction supplied it; for
    legacy callers, non-zero values and non-empty strings retain the previous
    behavior. Every problem has a stable code, field and severity so the API
    and review UI can revalidate after edits.
    """
    warnings: list[str] = []
    issues: list[ValidationIssue] = []

    def present(field: str, value: Any) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, datetime):
            return True
        if field in data.field_presence:
            return bool(data.field_presence[field])
        return float(value) != 0.0

    def add(field: str, code: str, message: str, severity: str = "warning", value: Any = None, expected: Any = None) -> None:
        issue = ValidationIssue(field=field, code=code, message=message, severity=severity, value=value, expected=expected)
        issues.append(issue)
        warnings.append(message)

    for field, label in (("timestamp", "ngày hóa đơn"), ("seller", "tên đơn vị bán"), ("total_cost", "tổng thanh toán")):
        if not present(field, getattr(data, field)):
            severity = "critical" if field == "total_cost" else "warning"
            add(field, "missing_required", f"Chưa xác định được {label}.", severity)
    if data.timestamp and data.timestamp.replace(tzinfo=None) > datetime.now():
        add("timestamp", "future_date", "Ngày hóa đơn nằm trong tương lai.", "error", data.timestamp.isoformat())

    def valid_tax_id(value: str | None) -> bool:
        if not value:
            return True
        compact = re.sub(r"[ .-]", "", value.upper())
        return bool(re.fullmatch(r"[0-9]{8,14}(-[0-9]{1,3})?", compact) or re.fullmatch(r"[0-9]{6,12}[A-Z][A-Z0-9-]{0,8}", compact))

    for field in ("tax_id", "buyer_tax_id"):
        value = getattr(data, field)
        if value and not valid_tax_id(value):
            add(field, "invalid_tax_id", f"Mã số thuế ở trường {field} không đúng định dạng.", "error", value)

    item_sum = 0.0
    for index, item in enumerate(data.items, 1):
        quantity_present = item.field_presence.get("quantity", item.quantity != 0)
        price_present = item.field_presence.get("unit_price", item.unit_price != 0)
        amount_present = item.field_presence.get("amount", item.amount != 0)
        if quantity_present and item.quantity < 0:
            add(f"items[{index}].quantity", "negative_quantity", f"Dòng {index} có số lượng âm.", "critical", item.quantity)
        if price_present and item.unit_price < 0:
            add(f"items[{index}].unit_price", "negative_unit_price", f"Dòng {index} có đơn giá âm.", "critical", item.unit_price)
        if amount_present and item.amount < 0:
            add(f"items[{index}].amount", "negative_amount", f"Dòng {index} có thành tiền âm.", "critical", item.amount)
        if quantity_present and price_present:
            expected_amount = round(item.quantity * item.unit_price, 2)
            actual_amount = item.amount if amount_present else expected_amount
            item_tolerance = max(0.01, abs(expected_amount) * tolerance_ratio)
            if amount_present and abs(actual_amount - expected_amount) > item_tolerance:
                add(
                    f"items[{index}].amount", "line_math_mismatch",
                    f"Dòng {index}: số lượng × đơn giá = {expected_amount:.2f}, nhưng thành tiền là {actual_amount:.2f}.",
                    "error", actual_amount, expected_amount,
                )
            item_sum += actual_amount
        elif amount_present:
            item_sum += item.amount
        elif quantity_present or price_present:
            add(f"items[{index}]", "incomplete_line", f"Dòng {index} thiếu số lượng hoặc đơn giá để kiểm tra.", "warning")

    item_sum = round(item_sum, 2)
    has_items = bool(data.items)
    subtotal_present = present("subtotal", data.subtotal)
    tax_present = present("tax", data.tax)
    discount_present = present("discount", data.discount)
    total_present = present("total_cost", data.total_cost)
    subtotal = item_sum if has_items else (data.subtotal if subtotal_present else 0.0)
    if has_items and subtotal_present and abs(item_sum - data.subtotal) > max(0.01, abs(item_sum) * tolerance_ratio):
        add("subtotal", "subtotal_mismatch", f"Tạm tính {data.subtotal:.2f} lệch tổng dòng hàng {item_sum:.2f}; dùng tổng dòng hàng để kiểm tra tổng.", "warning", data.subtotal, item_sum)
    if not subtotal_present and not has_items:
        add("subtotal", "missing_subtotal", "Thiếu tạm tính và không có dòng hàng để suy ra.", "error")

    if data.discount_rate is not None and subtotal > 0:
        rate_discount = round(subtotal * data.discount_rate / 100, 2)
        if discount_present and abs(data.discount - rate_discount) > max(0.02, abs(rate_discount) * 0.02):
            add("discount", "discount_rate_mismatch", f"Giảm giá {data.discount:.2f} không khớp tỷ lệ {data.discount_rate:g}% (dự kiến {rate_discount:.2f}).", "error", data.discount, rate_discount)
    if data.tax_included and data.tax_rate is not None and subtotal > 0 and tax_present:
        tax_from_inclusive = round(subtotal - subtotal / (1 + data.tax_rate / 100), 2)
        if abs(data.tax - tax_from_inclusive) > max(0.02, abs(tax_from_inclusive) * 0.02):
            add("tax", "tax_inclusive_mismatch", f"Tiền thuế không khớp cách tính giá đã gồm thuế (dự kiến {tax_from_inclusive:.2f}).", "error", data.tax, tax_from_inclusive)
    if data.tax < 0:
        add("tax", "negative_tax", "Tiền thuế không được âm.", "critical", data.tax)
    if data.discount < 0:
        add("discount", "negative_discount", "Giảm giá không được âm.", "critical", data.discount)
    if subtotal > 0 and data.discount > subtotal:
        add("discount", "discount_exceeds_subtotal", "Giảm giá lớn hơn tạm tính.", "error", data.discount, subtotal)

    rounding = data.rounding if present("rounding", data.rounding) else 0.0
    tax_component = 0.0 if data.tax_included else (data.tax if tax_present else 0.0)
    expected = round(subtotal + tax_component - (data.discount if discount_present else 0.0) + rounding, 2)
    if total_present:
        currency = (data.currency or "").upper()
        tolerance = max(0.01 if currency in {"VND", "JPY", "KRW"} else 0.02, abs(data.total_cost) * tolerance_ratio)
        if currency not in {"VND", "JPY", "KRW"}:
            tolerance = max(tolerance, 0.05 + abs(rounding))
        passed = abs(expected - data.total_cost) <= tolerance
        if not passed:
            add("total_cost", "total_mismatch", f"Tổng không khớp: dự kiến {expected:.2f}, hóa đơn ghi {data.total_cost:.2f} (dung sai {tolerance:.2f}).", "critical", data.total_cost, expected)
    else:
        passed = False
    if not total_present:
        add("total_cost", "missing_total", "Chưa rút trích được tổng thanh toán.", "critical")
    if confidence < threshold:
        add("confidence", "low_confidence", f"Độ tin cậy {confidence:.2f} thấp hơn ngưỡng {threshold:.2f}.", "warning", confidence, threshold)
    if not data.items:
        add("items", "missing_items", "Chưa phát hiện được dòng hàng hóa.", "warning")

    math = MathCheck(
        line_sum=item_sum if has_items else float(data.subtotal),
        tax=float(data.tax),
        discount=float(data.discount),
        expected_total=expected,
        actual_total=float(data.total_cost),
        delta=round(float(data.total_cost) - expected, 2),
        passed=bool(passed and not any(issue.severity in {"error", "critical"} for issue in issues)),
        tolerance=round(max(abs(float(data.total_cost) - expected), 0.01), 2),
    )
    status = "NEEDS_REVIEW" if issues else "VALIDATED"
    return ValidationResult(status, float(confidence), warnings, math, issues)
