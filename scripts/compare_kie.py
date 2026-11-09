"""Compare KIE extraction before vs. after the spatial-KIE rewrite.

Loads the BEFORE baseline (evaluation/fixtures/before_kie_results.json —
captured from the pre-rewrite extractor on the same real OCR token dumps),
re-runs the current extractor over ``evaluation/fixtures/ocr_tokens/*.json``
(real PaddleOCR output, no mocks), and scores both against the
hand-transcribed ground truth (evaluation/fixtures/kie_ground_truth.json).

Usage: python scripts/compare_kie.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.idp_engine.kie_extractor import KIEExtractor  # noqa: E402
from backend.app.idp_engine.ocr_detector import OCRToken  # noqa: E402
from backend.app.schemas.invoice import LineItem  # noqa: E402

FIXTURES = PROJECT_ROOT / "evaluation" / "fixtures"
TEXT_FIELDS = ("seller", "buyer", "address", "buyer_address", "tax_id", "buyer_tax_id", "invoice_id", "invoice_series", "invoice_form", "currency")
NUMBER_FIELDS = ("subtotal", "tax", "discount", "total_cost")
MONEY_TOL = 0.011


def load_tokens(name: str) -> list[OCRToken]:
    dump = json.loads((FIXTURES / "ocr_tokens" / f"{name}.json").read_text(encoding="utf-8"))
    rows = dump["tokens"] if isinstance(dump, dict) else dump
    return [OCRToken(t["text"], t["box"], t["confidence"]) for t in rows]


def norm_text(value: str | None) -> str:
    import unicodedata

    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    return " ".join(text.split())


def items_tuple(items: list[LineItem | dict]) -> list[tuple[str, float, float, float]]:
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append((norm_text(item.get("name")), float(item.get("quantity", 0)), float(item.get("unit_price", 0)), float(item.get("amount", 0))))
        else:
            rows.append((norm_text(item.name), float(item.quantity), float(item.unit_price), float(item.amount)))
    return rows


def field_hits(prediction: dict, truth: dict) -> tuple[list[str], list[str]]:
    """Return (correct_fields, wrong_fields) over all KIE entity fields."""
    correct: list[str] = []
    wrong: list[str] = []
    for name in TEXT_FIELDS:
        got, want = norm_text(prediction.get(name)), norm_text(truth.get(name))
        if got == want:
            correct.append(name)
        else:
            wrong.append(name)
    for name in NUMBER_FIELDS:
        got, want = float(prediction.get(name) or 0.0), float(truth.get(name) or 0.0)
        if abs(got - want) <= MONEY_TOL:
            correct.append(name)
        else:
            wrong.append(name)
    if norm_text(str(prediction.get("timestamp") or "")) == norm_text(str(truth.get("timestamp") or "")):
        correct.append("timestamp")
    else:
        wrong.append("timestamp")
    if items_tuple(prediction.get("items") or []) == items_tuple(truth.get("items") or []):
        correct.append("items")
    else:
        wrong.append("items")
    return correct, wrong


def main() -> int:
    truth = json.loads((FIXTURES / "kie_ground_truth.json").read_text(encoding="utf-8"))
    before = json.loads((FIXTURES / "before_kie_results.json").read_text(encoding="utf-8"))
    names = [path.stem for path in sorted((FIXTURES / "ocr_tokens").glob("*.json"))]

    after: dict[str, dict] = {}
    for name in names:
        data, _boxes, _raw, _details = KIEExtractor().extract_with_details(load_tokens(name))
        after[name] = {
            **{field: getattr(data, field) for field in TEXT_FIELDS},
            **{field: float(getattr(data, field) or 0.0) for field in NUMBER_FIELDS},
            "timestamp": data.timestamp.isoformat(timespec="seconds") if data.timestamp else None,
            "items": [
                {"name": i.name, "quantity": float(i.quantity), "unit_price": float(i.unit_price), "amount": float(i.amount)}
                for i in data.items
            ],
        }

    print(f"{'field':<14}" + "".join(f"{name:>28}" for name in names))
    all_fields = list(TEXT_FIELDS) + list(NUMBER_FIELDS) + ["timestamp", "items"]
    before_correct = after_correct = 0
    field_total = 0
    for field in all_fields:
        cells = []
        for name in names:
            want = truth.get(name, {}).get(field)
            before_ok = field_hits(before.get(name, {}), truth.get(name, {}))[0]
            after_ok, _ = field_hits(after.get(name, {}), truth.get(name, {}))
            b = "OK" if field in before_ok else "MISS"
            a = "OK" if field in after_ok else "MISS"
            if field in before_ok:
                before_correct += 1
            if field in after_ok:
                after_correct += 1
            cells.append(f"{b}→{a}")
        field_total += len(names)
        print(f"{field:<14}" + "".join(f"{cell:>28}" for cell in cells))
    print("-" * (14 + 28 * len(names)))
    print(f"entity accuracy: BEFORE {before_correct}/{field_total} = {before_correct / field_total:.1%}")
    print(f"                 AFTER  {after_correct}/{field_total} = {after_correct / field_total:.1%}")

    out = FIXTURES / "after_kie_results.json"
    out.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
