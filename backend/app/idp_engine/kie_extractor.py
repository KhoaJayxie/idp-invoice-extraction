from __future__ import annotations

from datetime import datetime
import re
import unicodedata

from rapidfuzz import fuzz

from ..schemas.invoice import InvoiceData, LineItem
from .ocr_detector import OCRToken
from .postprocessor import normalize_datetime, normalize_money, normalize_text


class KIEExtractor:
    """Lightweight layout-aware rule/regex KIE for Vietnamese invoices.

    This extractor is an explainable baseline, not a substitute for a trained
    LayoutLMv3/YOLO field classifier. Labels are retained on OCR boxes so the
    human reviewer can see which cues produced the structured fields.
    """

    LINE_Y_OVERLAP = 0.45

    LABELS: dict[str, tuple[str, ...]] = {
        "seller": ("tên cửa hàng", "ten cua hang", "đơn vị bán", "don vi ban", "người bán", "nguoi ban", "seller", "merchant"),
        "address": ("địa chỉ", "dia chi", "address"),
        "invoice_id": ("số hóa đơn", "so hoa don", "mã hóa đơn", "ma hoa don", "invoice no", "invoice number", "invoice id"),
        "timestamp": ("ngày giờ", "ngay gio", "ngày", "ngay", "date", "time"),
        "tax": ("thuế vat", "thue vat", "tiền thuế", "tien thue", "thuế", "thue", "vat", "tax"),
        "discount": ("giảm giá", "giam gia", "chiết khấu", "chiet khau", "discount"),
        "total_cost": ("tổng thanh toán", "tong thanh toan", "thành tiền", "thanh tien", "tổng cộng", "tong cong", "cần thanh toán", "can thanh toan", "total", "amount due"),
        "subtotal": ("cộng tiền hàng", "cong tien hang", "tạm tính", "tam tinh", "tổng phụ", "tong phu", "subtotal"),
    }
    # Prefix-anchored fuzzy match: OCR often drops vowels/letters on small
    # fonts ("Tổng cộng" -> "Tng cng"), so the exact substring check also
    # accepts near-matches only when the label is the leading part of a line.
    # 80 keeps real degraded labels (>=82) while rejecting look-alikes such
    # as "Huệ," vs "Thuế" (75).
    FUZZY_MIN_SCORE = 80.0
    FUZZY_MIN_LABEL_LEN = 4
    MONEY_PATTERN = re.compile(r"(?<![A-Za-z])[-(]?[0-9O][0-9O., ]*(?:\)|\s?(?:₫|đ|VND|VNĐ))?", re.IGNORECASE)
    ITEM_PATTERN = re.compile(
        r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s+([0-9Oo., ]+)\s+([0-9Oo., ]+)\s*(?:₫|đ|VND|VNĐ|d)?$",
        re.IGNORECASE,
    )
    # Name + quantity + total amount (no separate unit price column).
    ITEM_3FIELD_PATTERN = re.compile(
        r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s+([0-9Oo., ]+)\s*(?:₫|đ|VND|VNĐ|d)?$",
        re.IGNORECASE,
    )
    # Product descriptions often wrap or trail inside parentheses; a leading
    # or trailing "(...)" fragment belongs to the name, not the numbers.
    _PAREN_GROUP = re.compile(r"^\([^()]*\)\s*|\s*\([^()]*\)$")
    # Sanity bounds: receipt quantities and VND amounts. Rejects phone
    # numbers / account numbers misread as item rows ("+84 912 345 678").
    MAX_ITEM_QUANTITY = 999.0
    MIN_ITEM_MONEY = 500.0
    # Glued currency symbols ("3.000.000đ", OCR "3.000.000d") are stripped
    # before item matching; a letter right after the symbol means it belongs
    # to a word (e.g. "123 Đưng") and must be kept.
    _GLUED_CURRENCY = re.compile(
        r"(?<=\d)\s*(?:₫|đ|vnd|vnđ)(?![A-Za-zÀ-ỹ])|(?<=\d\.\d{3})d(?!\w)",
        re.IGNORECASE,
    )

    def extract(self, tokens: list[OCRToken]) -> tuple[InvoiceData, list[dict], str]:
        # Real detectors emit per-word/per-column boxes; rule KIE reasons about
        # whole visual lines ("name 2 3.000.000đ 6.000.000đ"). Reconstruct
        # lines by vertical overlap, then run the rules on those lines while
        # still emitting per-token boxes for the reviewer overlay.
        lines_tokens = self._merge_into_lines(tokens)
        lines = [normalize_text(token.text) for token in lines_tokens if normalize_text(token.text)]
        raw_text = "\n".join(lines)
        labels = self._label_tokens(lines_tokens)
        seller = self._extract_field(lines_tokens, "seller")
        if not seller:
            seller = self._first_non_label_line(lines_tokens)
        address = self._extract_field(lines_tokens, "address")
        invoice_id = self._extract_field(lines_tokens, "invoice_id")
        timestamp = self._extract_timestamp(lines_tokens, raw_text)
        tax = self._extract_money(lines_tokens, "tax")
        discount = self._extract_money(lines_tokens, "discount")
        total_cost = self._extract_money(lines_tokens, "total_cost")
        subtotal = self._extract_money(lines_tokens, "subtotal")
        items = self._extract_items(lines_tokens)
        if subtotal <= 0 and items:
            subtotal = sum(item.amount for item in items)

        token_confidence = self._confidence(tokens)
        field_confidence = {
            field: self._confidence([token])
            for field, token in labels.items()
            if token is not None
        }
        if seller and "seller" not in field_confidence:
            field_confidence["seller"] = token_confidence if seller else 0.0
        invoice = InvoiceData(
            invoice_id=invoice_id,
            seller=seller,
            address=address,
            timestamp=timestamp,
            subtotal=subtotal,
            tax=tax,
            discount=discount,
            total_cost=total_cost,
            items=items,
            field_confidence=field_confidence,
        )
        boxes = []
        for token in tokens:
            label = self._classify_line(token.text)
            boxes.append(
                {
                    "text": token.text,
                    "box": token.box,
                    "confidence": token.confidence,
                    "label": label,
                    "source": token.source,
                }
            )
        return invoice, boxes, raw_text

    def _merge_into_lines(self, tokens: list[OCRToken]) -> list[OCRToken]:
        """Group OCR tokens into visual lines using vertical overlap.

        Two tokens belong to the same line when their vertical spans overlap
        by at least LINE_Y_OVERLAP of the smaller height. Resulting line
        tokens carry the union box and the weakest member confidence, and are
        ordered top-to-bottom, left-to-right for the regex rules.
        """
        if not tokens:
            return []
        ordered = sorted(
            tokens,
            key=lambda token: ((token.box[1] + token.box[3]) / 2.0, token.box[0]),
        )
        groups: list[list[OCRToken]] = []
        for token in ordered:
            ty1, ty2 = float(token.box[1]), float(token.box[3])
            placed = False
            for group in groups:
                gy1 = min(float(item.box[1]) for item in group)
                gy2 = max(float(item.box[3]) for item in group)
                overlap = min(ty2, gy2) - max(ty1, gy1)
                reference = min(ty2 - ty1, gy2 - gy1)
                if reference > 0 and overlap / reference >= self.LINE_Y_OVERLAP:
                    group.append(token)
                    placed = True
                    break
            if not placed:
                groups.append([token])
        merged: list[OCRToken] = []
        for group in groups:
            group.sort(key=lambda token: token.box[0])
            text = " ".join(token.text for token in group).strip()
            box = [
                min(float(token.box[0]) for token in group),
                min(float(token.box[1]) for token in group),
                max(float(token.box[2]) for token in group),
                max(float(token.box[3]) for token in group),
            ]
            confidence = min(token.confidence for token in group)
            source = group[0].source
            merged.append(OCRToken(text, box, confidence, source))
        merged.sort(key=lambda token: (token.box[1], token.box[0]))
        return merged

    def _label_tokens(self, tokens: list[OCRToken]) -> dict[str, OCRToken | None]:
        return {field: self._find_label_token(tokens, field) for field in self.LABELS}

    def _extract_field(self, tokens: list[OCRToken], field: str) -> str | None:
        token = self._find_label_token(tokens, field)
        if token is None:
            return None
        value = self._value_after_label(token.text, self.LABELS[field])
        if value:
            if field == "invoice_id":
                match = re.search(r"[A-Z0-9][A-Z0-9./_-]{1,63}", value, re.IGNORECASE)
                return match.group(0).strip() if match else value[:128]
            return value.strip(" :.-")[:255]
        index = tokens.index(token)
        # Labels may appear alone on a line with the value directly beneath.
        if index + 1 < len(tokens) and field in ("seller", "address"):
            next_line = normalize_text(tokens[index + 1].text)
            if next_line and not self._classify_line(next_line):
                return next_line[:255]
        return None

    def _extract_timestamp(self, tokens: list[OCRToken], raw_text: str) -> datetime | None:
        labels = sorted(self.LABELS["timestamp"], key=len, reverse=True)
        for token in tokens:
            if self._label_score(labels, token.text) is None:
                continue
            value = self._value_after_label(token.text, labels)
            parsed = normalize_datetime(value or token.text)
            if parsed:
                return parsed
        return normalize_datetime(raw_text)

    def _extract_money(self, tokens: list[OCRToken], field: str) -> float:
        labels = self.LABELS[field]
        # Prefer the longest phrase, but also consider lines with noisy OCR labels.
        ordered = sorted(labels, key=len, reverse=True)
        for token in tokens:
            if self._label_score(ordered, token.text) is None:
                continue
            value = self._value_after_label(token.text, ordered)
            match = self.MONEY_PATTERN.search(value or "")
            if match:
                amount = normalize_money(match.group(0))
                if amount or re.search(r"\d", match.group(0)):
                    return amount
        # For unlabeled but standard footer layouts, use conservative aliases.
        if field == "total_cost":
            for token in reversed(tokens):
                folded = self._fold(token.text)
                if any(word in folded for word in ("thanh toan", "tong", "total", "amount due")):
                    values = self.MONEY_PATTERN.findall(token.text)
                    if values:
                        return normalize_money(values[-1])
        return 0.0

    def _extract_items(self, tokens: list[OCRToken]) -> list[LineItem]:
        items: list[LineItem] = []
        for token in tokens:
            text = normalize_text(token.text)
            if not text or self._classify_line(text) in {"TAX", "DISCOUNT", "TOTAL_COST", "SUBTOTAL", "TIMESTAMP", "INVOICE_ID", "SELLER", "ADDRESS"}:
                continue
            # Vietnamese receipts glue the currency symbol to amounts mid-line
            # ("3.000.000đ 3.000.000đ"); strip it so the numeric column regex
            # can see two clean numbers instead of failing on the 'đ'.
            text = self._GLUED_CURRENCY.sub("", text)
            item = self._match_item(text, token)
            if item is not None:
                items.append(item)
        return items

    def _match_item(self, text: str, token: OCRToken) -> LineItem | None:
        candidate = self._PAREN_GROUP.sub("", text).strip()
        match = self.ITEM_PATTERN.match(candidate) or self.ITEM_PATTERN.match(text)
        three_field = False
        if match:
            name, quantity_text, unit_price_text, amount_text = match.groups()
        else:
            match = self.ITEM_3FIELD_PATTERN.match(candidate) or self.ITEM_3FIELD_PATTERN.match(text)
            if not match:
                return None
            three_field = True
            name, quantity_text, amount_text = match.groups()
            unit_price_text = ""
        name = name.strip(" .:-")
        if len(name) < 2 or not re.search(r"[A-Za-zÀ-ỹ]", name):
            return None
        quantity = normalize_money(quantity_text)
        unit_price = normalize_money(unit_price_text)
        amount = normalize_money(amount_text)
        if quantity <= 0 or quantity > self.MAX_ITEM_QUANTITY:
            return None
        if three_field:
            if amount < self.MIN_ITEM_MONEY or quantity > 100:
                return None
            # A 3-field row implies amount = quantity × unit price; requiring
            # an (approximately) integral unit price rejects phone/account
            # numbers that happen to look like "name 345 678".
            unit_price = round(amount / quantity, 2) if quantity else 0.0
            if unit_price <= 0 or abs(unit_price - round(unit_price)) > max(1.0, unit_price * 0.01):
                return None
        else:
            if unit_price < self.MIN_ITEM_MONEY or amount < 0:
                return None
        return LineItem(
            name=name[:255],
            quantity=quantity,
            unit_price=unit_price,
            amount=amount if amount else quantity * unit_price,
            confidence=token.confidence,
            bbox=token.box,
        )

    def _find_label_token(self, tokens: list[OCRToken], field: str) -> OCRToken | None:
        labels = sorted(self.LABELS[field], key=len, reverse=True)
        for token in tokens:
            if self._label_score(labels, token.text) is not None:
                return token
        return None

    def _label_score(self, labels: tuple[str, ...] | list[str], text: str) -> float | None:
        """Score how strongly `text` contains one of `labels` at a word boundary.

        Exact accent-insensitive substring matches score ~100. Near-misses are
        accepted only against contiguous 1-4 word windows (so "Thuế" still
        matches the OCR-degraded "Thu (0%)" and "Tổng cộng" matches "Tng cng:",
        while an item name containing "thu" stays unclassified), using
        rapidfuzz on the folded strings.
        """
        folded = self._fold(text)
        words = folded.split()
        best: float | None = None
        for label in labels:
            needle = self._fold(label)
            index = folded.find(needle)
            if index >= 0:
                # Require a word boundary (or line start) before the label so
                # "thu" inside a longer word does not count.
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
                    if abs(len(window) - len(needle)) > 3:
                        continue
                    ratio = float(fuzz.ratio(needle, window))
                    if ratio >= self.FUZZY_MIN_SCORE:
                        best = ratio if best is None else max(best, ratio)
                        break
                if best is not None and best >= 100.0:
                    break
        return best

    def _first_non_label_line(self, tokens: list[OCRToken]) -> str | None:
        for token in tokens[:5]:
            text = normalize_text(token.text)
            if text and not self._classify_line(text) and re.search(r"[A-Za-zÀ-ỹ]", text):
                return text[:255]
        return None

    def _classify_line(self, text: str) -> str | None:
        # Avoid matching a generic 'ngày' substring in an address or item name:
        # _label_score anchors near-matches to the line start.
        for field, labels in self.LABELS.items():
            if self._label_score(labels, text) is not None:
                return field.upper()
        if self.ITEM_PATTERN.match(normalize_text(text)):
            return "LINE_ITEMS"
        return None

    def _value_after_label(self, text: str, labels: tuple[str, ...] | list[str]) -> str | None:
        """Return the part of `text` after its label.

        A colon separator is the most reliable cue on invoices and is used
        first, so OCR noise inside the label never eats into the value. Without
        a colon the label is stripped word by word (exact or fuzzy), which
        also copes with accent-dropping OCR.
        """
        parts = re.split(r"[:：]", text, maxsplit=1)
        if len(parts) == 2:
            value = parts[1].strip(" .-\t")
            return value or None
        words = text.split()
        for count in range(1, min(5, len(words))):
            head = " ".join(words[:count])
            scored_labels = sorted(labels, key=len, reverse=True)
            exact = any(self._fold(label) in self._fold(head) for label in scored_labels)
            fuzzy = exact or self._label_score(scored_labels, head) is not None
            if fuzzy:
                remainder = " ".join(words[count:]).strip(" .-\t")
                if remainder:
                    return remainder
        return None

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value.casefold())
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")

    @staticmethod
    def _confidence(tokens: list[OCRToken]) -> float:
        return min((max(0.0, min(1.0, token.confidence)) for token in tokens), default=0.0)
