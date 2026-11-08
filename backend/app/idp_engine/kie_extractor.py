from __future__ import annotations

"""Spatial layout-aware KIE for Vietnamese and English invoices/receipts.

Works on real OCR tokens (text + confidence + bbox):
tokens -> visual lines (y-overlap) -> x-gap segments (multi-column aware)
-> header / metadata / item-table / summary regions -> fields.

Each field reports the value, the raw OCR confidence of its evidence tokens,
and a separate KIE confidence that folds in how strong the structural
evidence was (exact label, spatial fallback, inference). Fields without
sufficient evidence stay None (null) instead of guessing.
"""

from dataclasses import dataclass, field as dc_field
from datetime import datetime
import re
import unicodedata
from typing import Any

from rapidfuzz import fuzz

from ..schemas.invoice import InvoiceData, LineItem
from .ocr_detector import OCRToken
from .postprocessor import currency_hint, normalize_datetime, normalize_money, normalize_text


# --------------------------------------------------------------------------
# Vocabularies — generic commerce terms (VN + EN), no invoice hardcoding.
# --------------------------------------------------------------------------

LABELS: dict[str, tuple[str, ...]] = {
    "seller": (
        "tên cửa hàng", "ten cua hang", "đơn vị bán", "don vi ban", "người bán", "nguoi ban",
        "seller", "supplier", "merchant",
    ),
    "buyer": (
        "hóa đơn được gửi cho", "hoa don duoc gui cho", "gửi cho", "gui cho", "đơn vị mua",
        "don vi mua", "người mua", "nguoi mua", "khách hàng", "khach hang", "buyer", "bill to",
        "billed to", "sold to", "customer",
    ),
    "address": ("địa chỉ", "dia chi", "address"),
    "invoice_id": (
        "document no", "document number", "doc no", "receipt no", "receipt number", "receipt #",
        "invoice no", "invoice number", "invoice #", "invoice id", "bill no", "bill number",
        "cash bill no", "số hóa đơn", "so hoa don", "mã hóa đơn", "ma hoa don",
        "số chứng từ", "so chung tu", "số phiếu", "so phieu", "ref no", "reference no",
    ),
    "timestamp": (
        "invoice date", "dated", "ngày giờ", "ngay gio", "ngày", "ngay", "date", "time",
        "thời gian", "thoi gian",
    ),
    "tax_id": (
        "mã số thuế", "ma so thue", "mst", "mã thuế", "ma thue", "tax id", "tax code", "tax no",
        "tax registration", "vat no", "vat id", "vat number", "gst no", "gst id", "gst number",
        "gst reg", "company reg", "co. reg", "business reg", "company registration",
    ),
    "subtotal": (
        "cộng tiền hàng", "cong tien hang", "tạm tính", "tam tinh", "tổng phụ", "tong phu",
        "sub total", "subtotal", "tiền hàng", "tien hang",
    ),
    "discount": ("giảm giá", "giam gia", "chiết khấu", "chiet khau", "discount"),
    "tax": (
        "thuế vat", "thue vat", "tiền thuế", "tien thue", "thuế", "thue", "total gst", "gst",
        "vat", "tax", "sst", "ppn", "sales tax",
    ),
    "total_cost": (
        "tổng thanh toán", "tong thanh toan", "thành tiền", "thanh tien", "tổng cộng", "tong cong",
        "grand total", "rounded total", "round total", "final total", "cần thanh toán",
        "can thanh toan", "total", "amount due", "tổng tiền", "tong tien",
    ),
}

TOTAL_MODIFIERS = ("grand", "rounded", "round total", "final")
_MONEY_LINE_EXCLUDE = re.compile(
    r"(?:qty|số lượng|so luong|discount|giảm|giam|sub\s?total|tạm tính|tam tinh|"
    r"tổng phụ|tong phu|rounding|adjustment|cash|change|tiền thối|tien thoi)",
    re.IGNORECASE,
)
_TAXID_LABEL_LINE = re.compile(
    r"(?:gst|tax|vat|mst|mã số thuế|ma so thue|mã thuế|ma thue|reg)", re.IGNORECASE
)
_TAXID_VALUE = re.compile(
    r"\b(?:\d{10}(?:-\d{3})?|\d{12,14}|[A-Za-z]{1,3}\d{5,9}-[A-Za-z]|\d{5,9}-[A-Za-z])\b"
)
_ID_VALUE_PATTERN = re.compile(r"(?=[A-Za-z0-9./#_-]*\d)[A-Za-z0-9][A-Za-z0-9./#_-]{2,63}")
_TIME_PATTERN = re.compile(r"\b(\d{1,2})[:h](\d{2})(?::(\d{2}))?\b")
_TIME_LABELS = ("time", "thời gian", "thoi gian")
_DATEISH_VALUE = re.compile(r"\d{1,2}\s*[/.\-]\s*\d{1,2}|\bth[áa]ng\b|\b20\d{2}\b")
_PHONE_LINE = re.compile(r"(?:\btel\b|\bphone\b|điện thoại|dien thoai|\bthoại\b|\bthoai\b|\bsdt\b|\bs\s?dt\b|\bfax\b|email|@|\bhp\b|\bhotline\b)", re.IGNORECASE)
_DATE_LINE = re.compile(r"(?:document|receipt|invoice|cash bill|member|salesperson|sales no|gst|tax|vat|mst|mã số|ma so|hóa đơn|hoa don|phiếu|phieu)", re.IGNORECASE)
_STREET_HINTS = (
    "jalan", "road", "street", "đường", "duong", "taman", "bandar", "kota", "thành phố",
    "thanh pho", "tp.", "quận", "quan ", "huyện", "huyen", "phường", "phuong", "ward",
    "district", "city", "province", "block", "lot", "no.",
)
_BUSINESS_HINTS = (
    "sdn bhd", "sendirian", "perniagaan", "perusahaan", "sdn.", "bhd", "trading", "enterprise",
    "company", "ltd", "limited", "llc", "inc", "corp", "group", "store", "mart", "minimart",
    "supermarket", "shop", "cửa hàng", "cua hang", "công ty", "cong ty", "tnhh", "cổ phần",
    "co phan", "nhà sách", "nha sach", "nhà xuất bản", "nha xuat ban", "tiệm", "tiem", "quán",
    "quan", "cafe", "café", "cà phê", "ca phe", "restaurant", "nhà hàng", "nha hang", "hotel",
    "khách sạn", "khach san", "pharmacy", "clinic", "bakery", "books", "stationery", "agency",
    "salon", "studio", "service", "services", "engineering", "electric", "electronic",
    "hardware", "motor", "auto", "fashion", "boutique", "printing", "construction",
)
_TABLE_ROLES: dict[str, tuple[str, ...]] = {
    "desc": ("item", "description", "desc", "code", "product", "sản phẩm", "san pham", "hàng hóa",
             "hang hoa", "hạng mục", "hang muc", "chi tiết", "chi tiet", "mô tả", "mo ta",
             "tên hàng", "ten hang", "goods", "particulars", "name"),
    "qty": ("qty", "quantity", "số lượng", "so luong", "sl"),
    "price": ("price", "unit", "đơn giá", "don gia", "giá", "gia", "rsp", "rate"),
    "amount": ("amount", "amt", "thành tiền", "thanh tien", "total", "tổng cộng", "tong cong",
               "tổng tiền", "tong tien", "sum"),
    "disc": ("disc", "discount", "giảm giá", "giam gia", "giảm", "giam"),
}
_SUMMARY_HINTS = (
    "sub total", "subtotal", "tạm tính", "tam tinh", "tổng phụ", "tong phu", "total",
    "tổng cộng", "tong cong", "tổng thanh toán", "thành tiền", "thanh tien", "amount due",
    "tot qty", "total qty", "tổng số lượng", "tong so luong", "rounding", "round",
    "adjustment", "cash", "change", "tiền mặt", "tien mat", "tiền thối", "tien thoi",
    "gst summary", "tax summary", "gst", "thuế", "thue", "vat", "tax", "discount",
    "giảm giá", "giam gia", "cần thanh toán", "can thanh toan",
)
_SUMMARY_LABELS = _SUMMARY_HINTS
_UNIT_WORDS = ("pc", "pcs", "unit", "units", "kg", "g", "gr", "ml", "box", "hộp", "hop", "cái",
               "cai", "set", "roll", "pkt")
METHOD_WEIGHTS = {
    "labeled_inline": 1.0,
    "labeled_right": 0.95,
    "labeled_below": 0.85,
    "computed": 0.8,
    "pattern_scan": 0.7,
    "business_inference": 0.5,
    "first_line_inference": 0.45,
    "address_run": 0.55,
}


# --------------------------------------------------------------------------
# Spatial primitives
# --------------------------------------------------------------------------

@dataclass(slots=True)
class _T:
    """OCR token with its index in the original boxes list."""
    idx: int
    tok: OCRToken

    @property
    def text(self) -> str:
        return self.tok.text

    @property
    def box(self) -> list[float]:
        return self.tok.box

    @property
    def conf(self) -> float:
        return self.tok.confidence

    @property
    def x1(self) -> float:
        return float(self.tok.box[0])

    @property
    def y1(self) -> float:
        return float(self.tok.box[1])

    @property
    def x2(self) -> float:
        return float(self.tok.box[2])

    @property
    def y2(self) -> float:
        return float(self.tok.box[3])


@dataclass(slots=True)
class _Seg:
    toks: list[_T]

    @property
    def x1(self) -> float:
        return min(t.x1 for t in self.toks)

    @property
    def x2(self) -> float:
        return max(t.x2 for t in self.toks)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.toks)


@dataclass(slots=True)
class _Line:
    toks: list[_T]
    segs: list[_Seg]
    y1: float
    y2: float

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segs)

    @property
    def box(self) -> list[float]:
        return [
            min(t.x1 for t in self.toks),
            self.y1,
            max(t.x2 for t in self.toks),
            self.y2,
        ]


@dataclass(slots=True)
class _Field:
    value: Any = None
    ocr_conf: float = 0.0
    kie_conf: float = 0.0
    method: str = ""
    box_ids: list[int] = dc_field(default_factory=list)

    def filled(self) -> bool:
        return self.value is not None and self.value != ""


@dataclass
class _Block:
    groups: list[list[_T]] = dc_field(default_factory=list)

    @property
    def toks(self) -> list[_T]:
        return [t for group in self.groups for t in group]

    @property
    def y1(self) -> float:
        return min(t.y1 for t in self.toks)

    @property
    def y2(self) -> float:
        return max(t.y2 for t in self.toks)

    @property
    def box(self) -> list[float]:
        toks = self.toks
        return [min(t.x1 for t in toks), min(t.y1 for t in toks), max(t.x2 for t in toks), max(t.y2 for t in toks)]


def _iter_block_tokens(blocks: list[list[list[_T]]]):
    for block in blocks:
        for group in block:
            yield from group


# --------------------------------------------------------------------------
# KIE engine
# --------------------------------------------------------------------------

class KIEExtractor:
    """Rule-based spatial KIE over real OCR tokens."""

    LABELS = LABELS  # expose the module vocabulary on the class (back-compat)
    LINE_Y_OVERLAP = 0.45
    SEGMENT_GAP_RATIO = 0.13
    SEGMENT_GAP_MIN = 110.0
    FUZZY_MIN_SCORE = 80.0
    FUZZY_MIN_LABEL_LEN = 4
    MONEY_PATTERN = re.compile(r"(?<![A-Za-z])[-(]?[0-9O][0-9O., ]*(?:\)|\s?(?:₫|đ|VND|VNĐ))?", re.IGNORECASE)
    ITEM_PATTERN = re.compile(r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s+([0-9Oo., ]+)\s+([0-9Oo., ]+)\s*(?:₫|đ|VND|VNĐ|d)?$", re.IGNORECASE)
    ITEM_3FIELD_PATTERN = re.compile(r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s+([0-9Oo., ]+)\s*(?:₫|đ|VND|VNĐ|d)?$", re.IGNORECASE)
    _PAREN_GROUP = re.compile(r"^\([^()]*\)\s*|\s*\([^()]*\)$")
    _GLUED_CURRENCY = re.compile(r"(?<=\d)\s*(?:₫|đ|vnd|vnđ)(?![A-Za-zÀ-ỹ])|(?<=\d\.\d{3})d(?!\w)", re.IGNORECASE)
    MAX_ITEM_QUANTITY = 999.0
    MIN_ITEM_MONEY = 500.0

    # ------------------------------------------------------------- lines

    def _build_lines(self, tokens: list[OCRToken]) -> list[_Line]:
        wrappers = [_T(i, tok) for i, tok in enumerate(tokens) if str(tok.text).strip()]
        ordered = sorted(wrappers, key=lambda t: ((t.y1 + t.y2) / 2.0, t.x1))
        page_width = max((t.x2 for t in wrappers), default=0.0) or 400.0
        gap_threshold = max(self.SEGMENT_GAP_RATIO * page_width, self.SEGMENT_GAP_MIN)
        groups: list[list[_T]] = []
        for token in ordered:
            ty1, ty2 = token.y1, token.y2
            placed = False
            for group in groups:
                gy1 = min(t.y1 for t in group)
                gy2 = max(t.y2 for t in group)
                overlap = min(ty2, gy2) - max(ty1, gy1)
                reference = min(ty2 - ty1, gy2 - gy1)
                if reference > 0 and overlap / reference >= self.LINE_Y_OVERLAP:
                    group.append(token)
                    placed = True
                    break
            if not placed:
                groups.append([token])
        lines: list[_Line] = []
        for group in groups:
            group.sort(key=lambda t: t.x1)
            segs: list[list[_T]] = [[group[0]]]
            for prev, token in zip(group, group[1:]):
                if token.x1 - prev.x2 >= gap_threshold:
                    segs.append([token])
                else:
                    segs[-1].append(token)
            seg_objs = [_Seg(segs[0])]
            for seg in segs[1:]:
                seg_objs.append(_Seg(seg))
            lines.append(_Line(group, seg_objs, min(t.y1 for t in group), max(t.y2 for t in group)))
        lines.sort(key=lambda line: (line.y1, line.box[0]))
        return lines

    # ----------------------------------------------------- label matching

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value.casefold())
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")

    def _label_score(self, labels: tuple[str, ...] | list[str], text: str) -> float | None:
        folded = self._fold(text)
        words = folded.split()
        best: float | None = None
        for label in labels:
            needle = self._fold(label)
            index = folded.find(needle)
            if index >= 0:
                before_ok = index == 0 or not folded[index - 1].isalnum()
                if before_ok:
                    score = 100.0 - min(index, 20) * 0.5
                    best = score if best is None else max(best, score)
                    continue
            if len(needle) < self.FUZZY_MIN_LABEL_LEN:
                continue
            for count in range(1, min(4, len(words)) + 1):
                for start in range(0, len(words) - count + 1):
                    window = " ".join(words[start : start + count])
                    # Short hints ("change", "gst") fuzzy-match only near-equal
                    # windows; longer labels allow 2 chars of OCR noise.
                    if abs(len(window) - len(needle)) > (1 if len(needle) <= 6 else 2):
                        continue
                    ratio = float(fuzz.ratio(needle, window))
                    if ratio >= self.FUZZY_MIN_SCORE:
                        best = ratio if best is None else max(best, ratio)
                        break
                if best is not None and best >= 100.0:
                    break
        return best

    def _value_after_label(self, text: str, labels: tuple[str, ...] | list[str]) -> str | None:
        parts = re.split(r"[:：]", text, maxsplit=1)
        if len(parts) == 2:
            value = parts[1].strip(" .-\t")
            return value or None
        words = text.split()
        for count in range(1, min(5, len(words))):
            head = " ".join(words[:count])
            scored_labels = sorted(labels, key=len, reverse=True)
            exact = any(self._fold(label) in self._fold(head) for label in scored_labels)
            if exact or self._label_score(scored_labels, head) is not None:
                remainder = " ".join(words[count:]).strip(" .-\t")
                if remainder:
                    return remainder
        return None

    def _classify_line(self, text: str) -> str | None:
        for field, labels in self.LABELS.items():
            if self._label_score(labels, text) is not None:
                return field.upper()
        if self.ITEM_PATTERN.match(normalize_text(text)) or self.ITEM_3FIELD_PATTERN.match(normalize_text(text)):
            return "LINE_ITEMS"
        return None

    # ------------------------------------------------------- number utils

    def _cell_number(self, text: str, decimal_dot: bool | None) -> float | None:
        """Parse a table/amount cell token into a float, tolerating OCR noise.

        9.000 (RM) -> 9.0 · GA 3 -> 3 · GA2 -> 2 · 1 PC -> 1 · 17.5C -> 17.5 ·
        Ođ -> 0 · CS00082258 / TD01167104 / 789417-W -> None (identifier-shaped).
        """
        raw = unicodedata.normalize("NFKC", str(text or "")).strip()
        raw = re.sub(r"(?i)\b(?:vnd|vnđ|myr|usd|sgd|rm)\b", "", raw)
        raw = re.sub(r"[₫$€£]", "", raw).strip()
        raw = re.sub(r"(?i)\b(?:pcs?|units?|kg|gr|ml|box|hộp|hop|cái|cai|set|roll|pkt)\b\.?", "", raw).strip()
        match = re.search(r"\d[\d.,]*\d|\d", raw)
        if match is None:
            # OCR reads 0 as O: a lone O/OO beside a currency marker is zero.
            if re.fullmatch(r"[Oo]{1,3}", raw):
                return 0.0
            return None
        prefix, suffix = raw[: match.start()], raw[match.end():]
        digits_only = re.sub(r"\D", "", match.group(0))
        if re.search(r"[A-Za-zÀ-ỹ]$", prefix):
            # Letters glued to the digits: accept only a 1-2 letter annotation
            # in front of a short number (GA2); CS00082258 / TD01167104 fail.
            if len(prefix) <= 2 and len(digits_only) <= 4:
                pass
            else:
                return None
        if re.search(r"[A-Za-zÀ-ỹ]{2,}", suffix):
            return None
        if re.search(r"-\s*[A-Za-z]\s*$", raw):
            return None  # 789417-W style registration numbers
        num = match.group(0).rstrip(".,")
        return normalize_money(num, decimal_dot)

    def _token_role(self, token: _T, columns: dict[str, tuple[float, float]]) -> str | None:
        center = (token.x1 + token.x2) / 2.0
        for role, (lo, hi) in columns.items():
            if lo <= center <= hi:
                return role
        best_role, best_dist = None, None
        for role, (lo, hi) in columns.items():
            dist = min(abs(center - lo), abs(center - hi))
            if best_dist is None or dist < best_dist:
                best_role, best_dist = role, dist
        return best_role

    def _role_keyword_match(self, token_text: str) -> set[str]:
        folded = self._fold(token_text)
        roles: set[str] = set()
        for role, keywords in _TABLE_ROLES.items():
            for keyword in keywords:
                needle = self._fold(keyword)
                if len(needle) <= 3:
                    if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", folded):
                        roles.add(role)
                        break
                elif needle in folded or (len(folded) >= 3 and fuzz.ratio(needle, folded) >= 85):
                    roles.add(role)
                    break
        return roles

    @staticmethod
    def _currency_decimal(text: str, hint: str | None) -> bool | None:
        detected = currency_hint(text) or hint
        if detected == "VND":
            return False
        if detected in {"MYR", "USD", "SGD", "EUR", "GBP"}:
            return True
        return None

    # ------------------------------------------------------- item table

    def _detect_table(self, lines: list[_Line]) -> tuple[int, int, dict[str, tuple[float, float]], str | None] | None:
        """Find the item-table header line(s).

        Returns (header_first, header_last, role -> x-range, currency).
        """
        for index, line in enumerate(lines):
            roles: dict[str, list[_T]] = {}
            for token in line.toks:
                for role in self._role_keyword_match(token.text):
                    roles.setdefault(role, []).append(token)
            if len(roles) < 2 or not ({"qty", "price", "amount"} & set(roles)):
                continue
            header_last = index
            for extra_index in range(index + 1, min(index + 3, len(lines))):
                extra_roles: set[str] = set()
                for token in lines[extra_index].toks:
                    extra_roles |= self._role_keyword_match(token.text)
                if extra_roles - set(roles):
                    for role in extra_roles:
                        roles.setdefault(role, []).extend(
                            t for t in lines[extra_index].toks if role in self._role_keyword_match(t.text)
                        )
                    header_last = extra_index
                else:
                    break
            ordered = sorted(
                ((role, min(t.x1 for t in toks), max(t.x2 for t in toks)) for role, toks in roles.items()),
                key=lambda item: item[1],
            )
            columns: dict[str, tuple[float, float]] = {}
            centers = {role: (x1 + x2) / 2.0 for role, x1, x2 in ordered}
            for position, (role, x1, x2) in enumerate(ordered):
                lo = x1 - 30.0
                hi = x2 + 30.0
                if position > 0:
                    lo = max(lo, (centers[ordered[position - 1][0]] + centers[role]) / 2.0)
                if position + 1 < len(ordered):
                    hi = min(hi, (centers[role] + centers[ordered[position + 1][0]]) / 2.0)
                columns[role] = (lo, hi)
            if "desc" not in columns:
                first_numeric = min(centers[role] for role in centers if role != "desc")
                columns["desc"] = (0.0, first_numeric - 40.0)
            header_text = " ".join(line.text for line in lines[index : header_last + 1])
            currency = currency_hint(header_text)
            return index, header_last, columns, currency
        return None

    def _parse_items(
        self, lines: list[_Line], header: tuple[int, int, dict[str, tuple[float, float]], str | None]
    ) -> tuple[list[LineItem], int, list[list[_T]]]:
        _, header_last, columns, currency = header
        decimal_dot = self._currency_decimal("", currency)
        pitch = self._line_pitch(lines)
        blocks: list[list[list[_T]]] = []
        open_block: list[list[_T]] | None = None
        summary_start = len(lines)
        for index in range(header_last + 1, len(lines)):
            line = lines[index]
            if self._is_summary_line(line.text):
                summary_start = index
                if open_block:
                    blocks.append(open_block)
                    open_block = None
                break
            groups = [list(seg.toks) for seg in line.segs]
            numeric_cells: dict[str, int] = {"qty": 0, "price": 0, "amount": 0}
            for seg in line.segs:
                for token in seg.toks:
                    role = self._token_role(token, columns)
                    value = self._cell_number(token.text, decimal_dot)
                    if value is not None and abs(value) >= 1e8:
                        value = None  # barcode/phone-shaped, never an item cell
                    if role in numeric_cells and value is not None:
                        numeric_cells[role] += 1
            has_numeric = any(numeric_cells.values())
            joins_open = open_block is not None and (line.y1 - self._block_y2(open_block)) <= 1.8 * pitch
            if has_numeric:
                if open_block is not None and not self._block_has_numeric(open_block, columns, decimal_dot) and joins_open:
                    open_block.extend(groups)
                else:
                    if open_block:
                        blocks.append(open_block)
                    open_block = groups
            elif joins_open:
                open_block.extend(groups)
            elif self._looks_like_junk(line.text):
                continue
            else:
                if open_block:
                    blocks.append(open_block)
                open_block = groups
        if open_block:
            blocks.append(open_block)
        # A wrapped numeric continuation ("unit price printed on the row
        # below") forms a desc-less block; fold it into the previous item
        # block so the price joins its row.
        merged: list[list[list[_T]]] = []
        for block in blocks:
            has_desc = any(re.search(r"[A-Za-zÀ-ỹ]", t.text) for group in block for t in group)
            if not has_desc and merged:
                merged[-1].extend(block)
            else:
                merged.append(block)
        items: list[LineItem] = []
        for block in merged:
            item = self._block_to_item(block, columns, decimal_dot)
            if item is not None:
                items.append(item)
        return items, summary_start, merged

    @staticmethod
    def _block_y2(block: list[list[_T]]) -> float:
        return max(t.y2 for group in block for t in group)

    @staticmethod
    def _line_pitch(lines: list[_Line]) -> float:
        tops = [line.y1 for line in lines]
        gaps = [b - a for a, b in zip(tops, tops[1:]) if b - a > 2]
        if not gaps:
            return 24.0
        gaps.sort()
        return max(8.0, gaps[len(gaps) // 2])

    @staticmethod
    def _looks_like_junk(text: str) -> bool:
        return not text.strip(" .:*-")

    def _is_summary_line(self, text: str) -> bool:
        # Fuzzy: OCR mangles labels ("Tổng cộng" -> "Tng cng"), so raw
        # substring matching would miss real summary lines.
        return self._label_score(_SUMMARY_LABELS, text) is not None

    def _block_has_numeric(self, block: list[list[_T]], columns: dict[str, tuple[float, float]], decimal_dot: bool | None) -> bool:
        for token in (t for group in block for t in group):
            role = self._token_role(token, columns)
            value = self._cell_number(token.text, decimal_dot)
            if role in {"qty", "price", "amount"} and value is not None and abs(value) < 1e8:
                return True
        return False

    def _block_to_item(
        self, block: list[list[_T]], columns: dict[str, tuple[float, float]], decimal_dot: bool | None
    ) -> LineItem | None:
        flat = [t for group in block for t in group]
        desc_toks: list[_T] = []
        cells: dict[str, list[tuple[_T, float]]] = {"qty": [], "price": [], "amount": [], "disc": []}
        desc_range = columns.get("desc")
        for token in flat:
            stripped = token.text.strip()
            role = self._token_role(token, columns)
            value = self._cell_number(token.text, decimal_dot)
            if value is not None and abs(value) >= 1e8:
                value = None  # barcode/phone-shaped, never an item cell
            if value is not None and (role == "desc" or (desc_range is not None and token.x2 <= desc_range[1] + 10)):
                # Numeric token in the description column: a bare 1-3 digit
                # integer is a quantity whose header column was lost to OCR
                # noise ("SÓ GIÒ" for "Số lượng"); longer digit runs are item
                # codes.
                if re.fullmatch(r"\d{1,3}", stripped):
                    cells["qty"].append((token, value))
                continue
            if role in cells and value is not None:
                cells[role].append((token, value))
            elif role in cells and len(stripped) <= 3:
                # OCR annotation junk ("GA", "*") inside a numeric column.
                continue
            elif re.search(r"[A-Za-zÀ-ỹ]", stripped) and not re.fullmatch(r"(?i)(?:rm|usd|myr|sgd)", stripped):
                desc_toks.append(token)
        while desc_toks and re.fullmatch(r"\d{1,13}", desc_toks[0].text.strip()):
            desc_toks.pop(0)
        name = normalize_text(" ".join(t.text for t in desc_toks))
        name = name.strip(" .:-")
        if len(name) < 2 or not re.search(r"[A-Za-zÀ-ỹ]", name):
            return None
        folded_name = self._fold(name)
        if any(hint in folded_name for hint in ("tot qty", "total qty", "tong so luong", "gst summary", "tax summary")):
            return None
        qty_values = [v for _, v in cells["qty"]]
        price_values = [v for _, v in cells["price"]]
        amount_values = [v for _, v in cells["amount"]]
        qty, price, amount, math_ok = self._resolve_row_numbers(qty_values, price_values, amount_values)
        if qty is None and price is None and amount is None:
            return None
        if amount is None and qty is not None and price is not None:
            amount = round(qty * price, 2)
            math_ok = True
        if price is None and qty and amount:
            price = round(amount / qty, 2)
        if qty is None:
            if price and amount and abs(amount / price - round(amount / price)) <= max(0.05, 0.01 * amount):
                qty = round(amount / price, 3)
            else:
                qty = 1.0
        if not (0 < qty <= 100000) or (price is not None and price < 0) or (amount is not None and amount < 0):
            return None
        ocr_conf = min(t.conf for t in flat)
        kie_conf = round(min(1.0, ocr_conf * (1.0 if math_ok else 0.9)), 3)
        box = [
            min(t.x1 for t in flat),
            min(t.y1 for t in flat),
            max(t.x2 for t in flat),
            max(t.y2 for t in flat),
        ]
        return LineItem(
            name=name[:255],
            quantity=float(qty),
            unit_price=float(price or 0.0),
            amount=float(amount or (float(qty) * float(price or 0.0))),
            confidence=round(ocr_conf, 3),
            bbox=box,
        )

    def _resolve_row_numbers(
        self, qty_values: list[float], price_values: list[float], amount_values: list[float]
    ) -> tuple[float | None, float | None, float | None, bool]:
        if qty_values and price_values and amount_values:
            best = None
            for q in qty_values:
                for p in price_values:
                    for a in amount_values:
                        error = abs(q * p - a)
                        tolerance = max(0.05, 0.02 * abs(a))
                        if error <= tolerance and (best is None or error < best[0]):
                            best = (error, q, p, a)
            if best is not None:
                _, qty, price, amount = best
                return qty, price, amount, True
        qty = qty_values[-1] if qty_values else None
        price = price_values[-1] if price_values else None
        amount = amount_values[-1] if amount_values else None
        return qty, price, amount, False

    # ------------------------------------------------- legacy line items

    def _legacy_items(self, lines: list[_Line], skip_from: int) -> list[LineItem]:
        """Regex fallback for receipts without a detectable table header."""
        items: list[LineItem] = []
        for index, line in enumerate(lines):
            if index >= skip_from:
                break
            text = normalize_text(line.text)
            if not text or self._classify_line(text) not in {None, "LINE_ITEMS"}:
                continue
            text = self._GLUED_CURRENCY.sub("", text)
            currency = currency_hint(text)
            decimal_dot = self._currency_decimal(text, currency)
            candidate = self._PAREN_GROUP.sub("", text).strip()
            match = self.ITEM_PATTERN.match(candidate) or self.ITEM_PATTERN.match(text)
            three_field = False
            if match:
                name_text, quantity_text, unit_price_text, amount_text = match.groups()
            else:
                match = self.ITEM_3FIELD_PATTERN.match(candidate) or self.ITEM_3FIELD_PATTERN.match(text)
                if not match:
                    continue
                three_field = True
                name_text, quantity_text, amount_text = match.groups()
                unit_price_text = ""
            name = name_text.strip(" .:-")
            if len(name) < 2 or not re.search(r"[A-Za-zÀ-ỹ]", name):
                continue
            quantity = normalize_money(quantity_text)
            unit_price = normalize_money(unit_price_text, decimal_dot)
            amount = normalize_money(amount_text, decimal_dot)
            if quantity <= 0 or quantity > self.MAX_ITEM_QUANTITY:
                continue
            if three_field:
                if amount < self.MIN_ITEM_MONEY or quantity > 100:
                    continue
                unit_price = round(amount / quantity, 2) if quantity else 0.0
                if unit_price <= 0 or abs(unit_price - round(unit_price)) > max(1.0, unit_price * 0.01):
                    continue
            else:
                if unit_price < self.MIN_ITEM_MONEY or amount < 0:
                    continue
            items.append(
                LineItem(
                    name=name[:255],
                    quantity=quantity,
                    unit_price=unit_price,
                    amount=amount if amount else quantity * unit_price,
                    confidence=line.toks[0].conf,
                    bbox=line.box,
                )
            )
        return items

    # ------------------------------------------------------ field helpers

    def _line_amount(self, line: _Line, label_seg_index: int, decimal_dot: bool | None) -> tuple[float | None, list[_T]]:
        """Rightmost numeric token at/after the label segment of the line."""
        candidates: list[tuple[int, _T, float]] = []
        for seg_index, seg in enumerate(line.segs):
            if seg_index < label_seg_index:
                continue
            for token in seg.toks:
                value = self._cell_number(token.text, decimal_dot) if _is_amountish(token.text) else None
                if value is not None:
                    candidates.append((seg_index, token, value))
        if not candidates:
            return None, []
        seg_index, token, value = candidates[-1]
        return value, [token]

    def _extract_money_field(
        self,
        lines: list[_Line],
        labels: tuple[str, ...],
        start: int,
        skip_lines: set[int] | None = None,
        stop_patterns: tuple[str, ...] = (),
        currency: str | None = None,
        prefer_modifier: bool = False,
        exclude_labels: tuple[tuple[str, tuple[str, ...]], ...] = (),
        exclude_regex: re.Pattern[str] | None = None,
    ) -> _Field | None:
        skip_lines = skip_lines or set()
        best: _Field | None = None
        for index in range(start, len(lines)):
            if index in skip_lines:
                continue
            line = lines[index]
            folded = self._fold(line.text)
            if any(pattern in folded for pattern in stop_patterns):
                break
            if exclude_regex is not None and exclude_regex.search(line.text):
                continue
            excluded = False
            for _name, exclude_label_set in exclude_labels:
                if self._label_score(exclude_label_set, line.text) is not None:
                    excluded = True
                    break
            if excluded:
                continue
            score = self._label_score(labels, line.text)
            if score is None:
                continue
            decimal_dot = self._currency_decimal(line.text, currency)
            label_seg = 0
            for seg_index, seg in enumerate(line.segs):
                if self._label_score(labels, seg.text) is not None:
                    label_seg = seg_index
                    break
            value, evidence = self._line_amount(line, label_seg, decimal_dot)
            if value is None:
                continue
            modifier = any(m in folded for m in TOTAL_MODIFIERS)
            ocr_conf = min(t.conf for t in evidence) if evidence else 0.0
            kie_conf = round(min(1.0, ocr_conf * METHOD_WEIGHTS["labeled_inline"]), 3)
            field = _Field(value, ocr_conf, kie_conf, "labeled_inline", [t.idx for t in evidence])
            if prefer_modifier:
                if best is None:
                    best = field
                    if modifier:
                        best.method = "labeled_modifier"
                else:
                    best_is_modifier = best.method == "labeled_modifier"
                    if (modifier and not best_is_modifier) or (
                        modifier == best_is_modifier and float(value) >= float(best.value)
                    ):
                        if modifier:
                            field.method = "labeled_modifier"
                        best = field
            else:
                if best is None:
                    best = field
        return best

    def _extract_labeled_text(
        self,
        lines: list[_Line],
        labels: tuple[str, ...],
        start: int = 0,
        end: int | None = None,
        value_pattern: re.Pattern[str] | None = None,
        allowed_methods: tuple[str, ...] = ("labeled_inline", "labeled_right", "labeled_below"),
    ) -> _Field | None:
        end = len(lines) if end is None else end
        for index in range(start, end):
            line = lines[index]
            for seg_index, seg in enumerate(line.segs):
                if self._label_score(labels, seg.text) is None:
                    continue
                # (1) same segment, text after the label
                raw_value = self._value_after_label(seg.text, labels)
                if raw_value and value_pattern:
                    found = value_pattern.search(raw_value)
                    raw_value = found.group(0).strip(" :.,-") if found else None
                if raw_value:
                    ocr_conf = min(t.conf for t in seg.toks)
                    weight = METHOD_WEIGHTS["labeled_inline"]
                    return _Field(raw_value, ocr_conf, round(min(1.0, ocr_conf * METHOD_WEIGHTS["labeled_inline"]), 3), "labeled_inline", [t.idx for t in seg.toks])
                # (2) next segment to the right on the same visual line
                if "labeled_right" in allowed_methods and seg_index + 1 < len(line.segs):
                    right = line.segs[seg_index + 1]
                    raw_value = right.text.strip(" :.-")
                    # A neighbouring label ("THANH TOÁN CHO:") is not a value.
                    if raw_value and self._classify_line(raw_value) is None:
                        if value_pattern:
                            found = value_pattern.search(raw_value)
                            if not found:
                                raw_value = None
                            else:
                                raw_value = found.group(0).strip(" :.,-")
                        if raw_value:
                            ocr_conf = min(t.conf for t in right.toks)
                            weight = METHOD_WEIGHTS["labeled_right"]
                            return _Field(raw_value, ocr_conf, round(min(1.0, ocr_conf * weight), 3), "labeled_right", [t.idx for t in right.toks])
                # (3) value directly below the label, in the same column
                if "labeled_below" in allowed_methods:
                    for below_index in range(index + 1, min(index + 4, end)):
                        below = lines[below_index]
                        if self._label_score(labels, below.text) is not None:
                            continue
                        if _DATE_LINE.search(below.text) and labels is not self.LABELS["address"]:
                            continue
                        overlap = min(seg.x2, below.box[2]) - max(seg.x1, below.box[0])
                        seg_width = max(1.0, seg.x2 - seg.x1)
                        if overlap / seg_width < 0.35:
                            continue
                        candidate_seg = below.segs[0]
                        raw_value = candidate_seg.text.strip(" :.-")
                        if not raw_value or self._classify_line(raw_value) is not None:
                            continue
                        if value_pattern:
                            found = value_pattern.search(raw_value)
                            if not found:
                                continue
                            raw_value = found.group(0).strip(" :.,-")
                        ocr_conf = min(t.conf for t in candidate_seg.toks)
                        weight = METHOD_WEIGHTS["labeled_below"]
                        return _Field(raw_value, ocr_conf, round(min(1.0, ocr_conf * weight), 3), "labeled_below", [t.idx for t in candidate_seg.toks])
        return None

    # ----------------------------------------------------- party fields

    def _business_score(self, text: str) -> float:
        folded = self._fold(text)
        hints = sum(1 for hint in _BUSINESS_HINTS if hint in folded)
        letters = [c for c in text if c.isalpha()]
        caps = sum(1 for c in letters if c.isupper()) / max(1, len(letters)) if letters else 0.0
        score = hints * 2.0 + (1.0 if caps >= 0.6 else 0.0)
        if "(" in text and ")" in text:
            score += 0.5
        return score

    def _line_index_of(self, lines: list[_Line], token_idx: int) -> int | None:
        for index, line in enumerate(lines):
            if any(t.idx == token_idx for t in line.toks):
                return index
        return None

    def _extract_labeled_line(self, lines: list[_Line], labels: tuple[str, ...], start: int, end: int) -> tuple[int, int] | None:
        """First (line_index, seg_index) whose segment carries the label."""
        for index in range(start, end):
            for seg_index, seg in enumerate(lines[index].segs):
                if self._label_score(labels, seg.text) is not None:
                    return index, seg_index
        return None

    def _seller_from_header(self, lines: list[_Line], end: int) -> tuple[_Field, set[int]] | None:
        """Business-name inference for the merchant header block.

        Picks the topmost line with a strong business-name signal, keeps
        digit-free continuation lines (OCR splits a long shop name), and
        ignores label-bearing segments (a date on the same visual line).
        """
        pitch = self._line_pitch(lines)
        chosen: tuple[float, _Field, set[int]] | None = None
        for index in range(end):
            line = lines[index]
            folded = self._fold(line.text)
            if any(h in folded for h in ("document", "receipt", "invoice", "cash bill", "cashier", "member", "salesperson")):
                continue
            name_segs: list[_Seg] = []
            for seg in line.segs:
                if self._label_score(self.LABELS["timestamp"], seg.text) is not None:
                    continue
                if re.search(r"\d", seg.text):
                    continue
                name_segs.append(seg)
            if not name_segs:
                continue
            seg_text = " ".join(seg.text for seg in name_segs)
            if not re.search(r"[A-Za-zÀ-ỹ]", seg_text):
                continue
            score = self._business_score(seg_text)
            if score < 2.0:
                continue
            # The merchant header is the TOPMOST strong business line — deeper
            # company-looking lines (buyer blocks) must not win.
            name_parts = [normalize_text(seg_text)]
            box_ids = [t.idx for seg in name_segs for t in seg.toks]
            used = {index}
            confs = [t.conf for seg in name_segs for t in seg.toks]
            cursor = line
            for nxt in range(index + 1, min(index + 3, end)):
                nxt_line = lines[nxt]
                if nxt_line.y1 - cursor.y2 > max(30.0, 1.2 * (cursor.y2 - cursor.y1)):
                    break
                nxt_folded = self._fold(nxt_line.text)
                if re.search(r"\d", nxt_line.text):
                    break
                if self._label_score(self.LABELS["timestamp"], nxt_line.text) is not None:
                    continue
                if any(h in nxt_folded for h in ("document", "receipt", "invoice", "cashier", "member")):
                    continue
                if self._business_score(nxt_line.text) >= 2.0:
                    break
                name_parts.append(normalize_text(nxt_line.text))
                box_ids.extend(t.idx for t in nxt_line.toks)
                confs.extend(t.conf for t in nxt_line.toks)
                used.add(nxt)
                cursor = nxt_line
            name = normalize_text(" ".join(name_parts))
            ocr_conf = min(confs)
            fld = _Field(name, ocr_conf, round(min(1.0, ocr_conf * METHOD_WEIGHTS["business_inference"]), 3), "business_inference", box_ids)
            if chosen is None:
                chosen = (score, fld, used)
                break
        if chosen is None:
            return None
        _score, fld, used = chosen
        return fld, used

    def _extract_buyer_fallback(self, lines: list[_Line], end: int, seller_idx: int | None, used_lines: set[int]) -> _Field | None:
        pitch = self._line_pitch(lines)
        for index in range(max(0, seller_idx or 0) + 1, end):
            if index in used_lines:
                continue
            line = lines[index]
            folded = self._fold(line.text)
            if any(h in folded for h in ("document", "receipt", "invoice", "cash bill", "cashier", "member", "gst", "tax", "vat", "mst", "date", "ngày", "ngay")):
                continue
            if re.search(r"\d", line.text) or _PHONE_LINE.search(line.text):
                continue
            if self._business_score(line.text) < 2.0:
                continue
            if seller_idx is not None and line.y1 - lines[seller_idx].y2 > 12 * pitch:
                continue
            ocr_conf = min(t.conf for t in line.toks)
            return _Field(
                normalize_text(line.text),
                ocr_conf,
                round(min(1.0, ocr_conf * METHOD_WEIGHTS["business_inference"]), 3),
                "business_inference",
                [t.idx for t in line.toks],
            )
        return None

    # ------------------------------------------------------- run fields

    def _extract_address_run(
        self,
        lines: list[_Line],
        end: int,
        anchor_idx: int | None,
        anchor_x: tuple[float, float] | None = None,
    ) -> _Field | None:
        """Collect the address block below a party (seller/buyer) line.

        `anchor_x` pins the party's column: on two-column headers the value
        segment's x-range keeps the run inside its own column instead of the
        widest (possibly right-hand) segment of the anchor line.
        """
        if anchor_idx is None or anchor_idx + 1 >= end:
            return None
        anchor = lines[anchor_idx]
        if anchor_x is not None:
            ax1, ax2 = anchor_x
        else:
            anchor_seg = max(anchor.segs, key=lambda s: s.x2 - s.x1)
            ax1, ax2 = anchor_seg.x1, anchor_seg.x2
        pitch = self._line_pitch(lines)
        parts: list[str] = []
        confs: list[float] = []
        ids: list[int] = []
        consecutive_skip = 0
        for index in range(anchor_idx + 1, min(anchor_idx + 8, end)):
            line = lines[index]
            if line.y1 - lines[index - 1].y2 > 2.5 * pitch:
                break
            # Keep only the segments inside the party's column.
            if len(line.segs) > 1:
                segs = [
                    s for s in line.segs
                    if (min(ax2, s.x2) - max(ax1, s.x1)) >= 0.3 * max(1.0, s.x2 - s.x1)
                ] or list(line.segs)
            else:
                segs = list(line.segs)
            text = " ".join(s.text for s in segs)
            folded = self._fold(text)
            # Phone/tax-id lines and receipt- or invoice-number lines (OCR may
            # mangle the label: "Recelpt #: CS...") are never address parts.
            if (
                _PHONE_LINE.search(text)
                or _TAXID_LABEL_LINE.search(text)
                or _TAXID_VALUE.search(text)
                or self._label_score(self.LABELS["invoice_id"], text) is not None
            ):
                consecutive_skip += 1
                if consecutive_skip >= 2:
                    break
                continue
            if self._label_score(self.LABELS["timestamp"], text) is not None or _DATE_LINE.search(text):
                break
            if self._is_summary_line(text):
                break
            overlap = min(ax2, line.box[2]) - max(ax1, line.box[0])
            if overlap / max(1.0, ax2 - ax1) < 0.3:
                consecutive_skip += 1
                if consecutive_skip >= 2:
                    break
                continue
            if not (re.search(r"\d", text) or any(h in folded for h in _STREET_HINTS) or parts):
                break
            parts.append(normalize_text(" ".join(s.text.rstrip(" ,") for s in segs)))
            confs.extend(t.conf for s in segs for t in s.toks)
            ids.extend(t.idx for s in segs for t in s.toks)
            consecutive_skip = 0
        if not parts:
            return None
        text = normalize_text(", ".join(parts))
        if not re.search(r"\d", text) and not any(h in self._fold(text) for h in _STREET_HINTS):
            return None
        ocr_conf = min(confs)
        return _Field(text, ocr_conf, round(min(1.0, ocr_conf * METHOD_WEIGHTS["address_run"]), 3), "address_run", ids)

    def _extract_timestamp_field(self, lines: list[_Line], end: int) -> _Field | None:
        date_labels = tuple(l for l in self.LABELS["timestamp"] if l not in ("time", "thời gian", "thoi gian"))
        for index in range(end):
            line = lines[index]
            for seg_index, seg in enumerate(line.segs):
                if self._label_score(date_labels, seg.text) is None:
                    continue
                raw = self._value_after_label(seg.text, date_labels)
                if not raw and seg_index + 1 < len(line.segs):
                    raw = line.segs[seg_index + 1].text.strip(" :.-")
                if not raw:
                    continue
                value_text = raw
                # A single OCR block like "Date. 09/02/2018 Time:08:32:00"
                # splits at the clock's colon, so the raw value can hold only
                # the time. When the raw value carries no date, rebuild it
                # from the whole segment and fold a Time/Thời gian marker
                # into a plain separator.
                if not _DATEISH_VALUE.search(value_text):
                    rebuilt = re.sub(
                        r"^.*?\b(?:date|dated|ng[àa]y|ngay)\b\s*[:.\-]?\s*",
                        "", seg.text, count=1, flags=re.IGNORECASE,
                    )
                    rebuilt = re.sub(
                        r"\b(?:time|th\w*\s*gian|gi[ơo])\s*[:.]?\s*",
                        " ", rebuilt, flags=re.IGNORECASE,
                    ).strip(" :.-")
                    if rebuilt and _DATEISH_VALUE.search(rebuilt):
                        value_text = rebuilt
                if not _TIME_PATTERN.search(value_text):
                    for j in range(max(0, index - 3), min(end, index + 4)):
                        if j == index:
                            # clock in a sibling segment of the same visual line
                            for other_seg in line.segs:
                                if other_seg is seg:
                                    continue
                                if self._label_score(_TIME_LABELS, other_seg.text) is not None:
                                    m = _TIME_PATTERN.search(other_seg.text)
                                    if m:
                                        value_text = f"{value_text} {m.group(0).replace('h', ':', 1)}"
                                    break
                            continue
                        other = lines[j]
                        if self._label_score(_TIME_LABELS, other.text) is not None:
                            m = _TIME_PATTERN.search(other.text)
                            if m:
                                value_text = f"{value_text} {m.group(0).replace('h', ':', 1)}"
                            break
                parsed = normalize_datetime(value_text) or normalize_datetime(seg.text)
                if parsed:
                    ocr_conf = min(t.conf for t in seg.toks)
                    method = "labeled_inline" if self._value_after_label(seg.text, date_labels) else "labeled_right"
                    return _Field(parsed, ocr_conf, round(min(1.0, ocr_conf * METHOD_WEIGHTS[method]), 3), method, [t.idx for t in seg.toks])
        for index in range(end):
            parsed = normalize_datetime(lines[index].text)
            if parsed is not None:
                toks = lines[index].toks
                return _Field(parsed, min(t.conf for t in toks), 0.7, "pattern_scan", [t.idx for t in toks])
        return None

    def _extract_currency(self, lines: list[_Line]) -> _Field | None:
        counts: dict[str, int] = {}
        evidence: dict[str, list[_T]] = {}
        for line in lines:
            for token in line.toks:
                code = currency_hint(token.text)
                if code:
                    counts[code] = counts.get(code, 0) + 1
                    evidence.setdefault(code, []).append(token)
        if not counts:
            return None
        code = max(counts, key=lambda k: (counts[k], -min(t.y1 for t in evidence[k])))
        toks = evidence[code]
        ocr_conf = min(t.conf for t in toks)
        return _Field(code, ocr_conf, round(min(1.0, ocr_conf * 0.9), 3), "pattern_scan", [t.idx for t in toks][:8])

    def _extract_tax_id_pattern(self, lines: list[_Line], end: int, skip_lines: set[int]) -> _Field | None:
        for index in range(end):
            if index in skip_lines:
                continue
            line = lines[index]
            if _PHONE_LINE.search(line.text):
                continue
            found = _TAXID_VALUE.search(line.text)
            if found:
                ocr_conf = min(t.conf for t in line.toks)
                return _Field(found.group(0), ocr_conf, round(min(1.0, ocr_conf * METHOD_WEIGHTS["pattern_scan"]), 3), "pattern_scan", [t.idx for t in line.toks])
        return None

    def _first_summary_line(self, lines: list[_Line]) -> int:
        for index, line in enumerate(lines):
            if _TAXID_LABEL_LINE.search(line.text):
                # "GST Reg No: 789417-W"/"MST: 0311234567" are identifier
                # lines — their amount-ish tokens all match the tax-id value
                # shape, so they must not start the money summary. A
                # taxid-labeled line with a real amount ("Thuế VAT: 8.500")
                # still does.
                amountish = [t.text for t in line.toks if _is_amountish(t.text)]
                if not amountish or all(_TAXID_VALUE.search(v) for v in amountish):
                    continue
            if self._is_summary_line(line.text):
                return index
        return len(lines)

    # --------------------------------------------------------- orchestration

    def extract_with_details(
        self, tokens: list[OCRToken], currency: str | None = None
    ) -> tuple[InvoiceData, list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
        lines = self._build_lines(tokens)
        header = self._detect_table(lines)
        items: list[LineItem] = []
        summary_start = len(lines)
        if header:
            items, summary_start, _blocks = self._parse_items(lines, header)
            if not items:
                items = self._legacy_items(lines, summary_start)
        else:
            summary_start = self._first_summary_line(lines)
            items = self._legacy_items(lines, summary_start)
        # Address blocks live between the party headers and the item table;
        # never let an address run wander into the goods grid.
        addr_end = min(summary_start, header[0]) if header else summary_start

        # --- seller -------------------------------------------------------
        seller_fld = self._extract_labeled_text(
            lines, self.LABELS["seller"], 0, summary_start, allowed_methods=("labeled_inline", "labeled_right")
        )
        seller_used: set[int] = set()
        seller_idx: int | None = None
        if seller_fld:
            seller_idx = self._line_index_of(lines, seller_fld.box_ids[0]) if seller_fld.box_ids else None
            if seller_idx is not None:
                seller_used.add(seller_idx)
        if seller_fld is None:
            inferred = self._seller_from_header(lines, summary_start)
            if inferred:
                seller_fld, seller_used = inferred
                seller_idx = min(seller_used) if seller_used else None

        # --- buyer --------------------------------------------------------
        buyer_fld = self._extract_labeled_text(
            lines, self.LABELS["buyer"], 0, summary_start, allowed_methods=("labeled_inline", "labeled_right", "labeled_below")
        )
        buyer_idx: int | None = None
        if buyer_fld and buyer_fld.box_ids:
            buyer_idx = self._line_index_of(lines, buyer_fld.box_ids[0])
        if buyer_fld is None:
            buyer_fld = self._extract_buyer_fallback(lines, summary_start, seller_idx, seller_used)
            if buyer_fld and buyer_fld.box_ids:
                buyer_idx = self._line_index_of(lines, buyer_fld.box_ids[0])

        # --- addresses ----------------------------------------------------
        def value_xrange(fld: _Field | None) -> tuple[float, float] | None:
            """x-range of the segment that produced `fld` (column anchor)."""
            if not fld or not fld.box_ids:
                return None
            want = set(fld.box_ids)
            for line in lines:
                for seg in line.segs:
                    if any(t.idx in want for t in seg.toks):
                        return (seg.x1, seg.x2)
            return None

        address_labeled = self._extract_labeled_text(
            lines, self.LABELS["address"], 0, summary_start, allowed_methods=("labeled_inline", "labeled_right", "labeled_below")
        )
        address_fld = address_labeled or self._extract_address_run(lines, addr_end, seller_idx, value_xrange(seller_fld))
        buyer_address_fld = (
            self._extract_address_run(lines, addr_end, buyer_idx, value_xrange(buyer_fld))
            if buyer_idx is not None
            else None
        )

        # --- identifiers ----------------------------------------------------
        invoice_id_fld = self._extract_labeled_text(
            lines, self.LABELS["invoice_id"], 0, summary_start, value_pattern=_ID_VALUE_PATTERN, allowed_methods=("labeled_inline", "labeled_right")
        )
        tax_id_fld = self._extract_labeled_text(
            lines, self.LABELS["tax_id"], 0, summary_start, value_pattern=_TAXID_VALUE, allowed_methods=("labeled_inline", "labeled_right")
        )
        if tax_id_fld is None:
            scan_end = min(seller_idx + 6, summary_start) if seller_idx is not None else min(6, summary_start)
            tax_id_fld = self._extract_tax_id_pattern(lines, scan_end, seller_used)
        buyer_tax_id_fld = None
        if buyer_idx is not None:
            buyer_tax_id_fld = self._extract_labeled_text(
                lines, self.LABELS["tax_id"], buyer_idx + 1, min(buyer_idx + 6, summary_start), value_pattern=_TAXID_VALUE,
                allowed_methods=("labeled_inline", "labeled_right", "labeled_below"),
            )
            if buyer_tax_id_fld and tax_id_fld and str(buyer_tax_id_fld.value) == str(tax_id_fld.value):
                buyer_tax_id_fld = None

        timestamp_fld = self._extract_timestamp_field(lines, summary_start)
        currency_fld = self._extract_currency(lines)
        currency_seen = currency or (str(currency_fld.value) if currency_fld else None)

        # --- money fields (summary region) ---------------------------------
        subtotal_fld = self._extract_money_field(lines, self.LABELS["subtotal"], start=summary_start, currency=currency_seen)
        tax_fld = self._extract_money_field(
            lines,
            self.LABELS["tax"],
            start=summary_start,
            currency=currency_seen,
            exclude_labels=(("#taxid", self.LABELS["tax_id"]), ("#subtotal", self.LABELS["subtotal"])),
        )
        discount_fld = self._extract_money_field(lines, self.LABELS["discount"], start=summary_start, currency=currency_seen)
        total_fld = self._extract_money_field(
            lines,
            self.LABELS["total_cost"],
            start=summary_start,
            currency=currency_seen,
            prefer_modifier=True,
            exclude_labels=(("#subtotal", self.LABELS["subtotal"]), ("#tax", self.LABELS["tax"]), ("#discount", self.LABELS["discount"])),
            exclude_regex=_MONEY_LINE_EXCLUDE,
            stop_patterns=("gst summary", "tax summary", "tổng hợp thuế", "tong hop thue"),
        )
        fields: dict[str, _Field | None] = {
            "invoice_id": invoice_id_fld,
            "seller": seller_fld,
            "buyer": buyer_fld,
            "address": address_fld,
            "buyer_address": buyer_address_fld,
            "tax_id": tax_id_fld,
            "buyer_tax_id": buyer_tax_id_fld,
            "currency": currency_fld,
            "timestamp": timestamp_fld,
            "subtotal": subtotal_fld,
            "tax": tax_fld,
            "discount": discount_fld,
            "total_cost": total_fld,
        }

        def text_of(fld: _Field | None) -> str | None:
            if fld is None or fld.value is None or fld.value == "":
                return None
            return normalize_text(str(fld.value))

        invoice = InvoiceData(
            invoice_id=text_of(invoice_id_fld),
            seller=text_of(seller_fld),
            buyer=text_of(buyer_fld),
            address=text_of(address_fld),
            buyer_address=text_of(buyer_address_fld),
            tax_id=text_of(tax_id_fld),
            buyer_tax_id=text_of(buyer_tax_id_fld),
            currency=text_of(currency_fld),
            timestamp=timestamp_fld.value if timestamp_fld and isinstance(timestamp_fld.value, datetime) else None,
            subtotal=float(subtotal_fld.value) if subtotal_fld and subtotal_fld.filled() else 0.0,
            tax=float(tax_fld.value) if tax_fld and tax_fld.filled() else 0.0,
            discount=float(discount_fld.value) if discount_fld and discount_fld.filled() else 0.0,
            total_cost=float(total_fld.value) if total_fld and total_fld.filled() else 0.0,
            items=items,
            field_confidence={},
        )
        if invoice.subtotal <= 0 and items:
            sums = round(sum(item.amount for item in items), 2)
            invoice.subtotal = sums
            subtotal_fld = _Field(sums, min((i.confidence for i in items), default=0.0), 0.8, "computed", [])
            fields["subtotal"] = subtotal_fld
        field_confidence: dict[str, float] = {}
        field_details: dict[str, dict[str, Any]] = {}
        for name, fld in fields.items():
            if fld is None or not fld.filled():
                continue
            field_confidence[name] = fld.kie_conf
            value_out = fld.value
            if isinstance(value_out, datetime):
                value_out = value_out.isoformat(timespec="seconds")
            field_details[name] = {
                "value": value_out,
                "ocr_conf": round(fld.ocr_conf, 3),
                "kie_conf": fld.kie_conf,
                "method": fld.method,
                "box_ids": fld.box_ids,
            }
        if items:
            field_details["items"] = {
                "value": len(items),
                "ocr_conf": round(min(i.confidence for i in items), 3),
                "kie_conf": round(sum(i.confidence for i in items) / len(items) * 0.95, 3),
                "method": "table_rows" if header else "line_regex",
                "count": len(items),
            }
        invoice.field_confidence = field_confidence

        boxes: list[dict[str, Any]] = []
        for token in tokens:
            boxes.append(
                {
                    "text": token.text,
                    "box": token.box,
                    "confidence": token.confidence,
                    "label": self._classify_line(token.text),
                    "source": token.source,
                }
            )
        raw_text = "\n".join(line.text for line in lines)
        return invoice, boxes, raw_text, field_details

    def extract(self, tokens: list[OCRToken]) -> tuple[InvoiceData, list[dict[str, Any]], str]:
        invoice, boxes, raw_text, _details = self.extract_with_details(tokens)
        return invoice, boxes, raw_text


def _is_amountish(text: str) -> bool:
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    digits = re.sub(r"[^0-9]", "", folded)
    return bool(digits) and len(digits) >= 2


def extract(tokens: list[OCRToken]) -> tuple[InvoiceData, list[dict[str, Any]], str]:
    """Backwards-compatible 3-tuple wrapper over extract_with_details."""
    return KIEExtractor().extract(tokens)
