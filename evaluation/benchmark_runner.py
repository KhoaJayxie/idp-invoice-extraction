"""Reproducible P7 evaluation runner.

The default benchmark replays the captured *real OCR token dumps* through the
current Spatial KIE extractor. It never presents replay as a fresh OCR run.
Live VLM/Hybrid runs are opt-in and are marked unavailable when the manifest's
images or the configured local VLM are not present.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.idp_engine.kie_extractor import KIEExtractor
from backend.app.idp_engine.ocr_detector import OCRToken
from backend.app.idp_engine.postprocessor import validate_invoice
from backend.app.schemas.invoice import InvoiceData
from evaluation.evaluate_metrics import cer, wer

FIELDS = (
    "seller", "buyer", "address", "buyer_address", "tax_id", "buyer_tax_id",
    "invoice_id", "invoice_series", "invoice_form", "currency", "timestamp",
    "subtotal", "tax", "discount", "total_cost",
)
NUMERIC_FIELDS = {"subtotal", "tax", "discount", "total_cost", "quantity", "unit_price", "amount"}


def _norm(value: Any) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _same(field: str, expected: Any, predicted: Any) -> bool:
    if not _present(expected) or not _present(predicted):
        return not _present(expected) and not _present(predicted)
    if field in NUMERIC_FIELDS:
        try:
            left, right = float(expected), float(predicted)
            return abs(left - right) <= max(0.011, abs(left) * 0.001)
        except (TypeError, ValueError):
            return False
    if field == "timestamp":
        return _norm(expected)[:19] == _norm(predicted)[:19]
    return _norm(expected) == _norm(predicted)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truth_path_and_key(value: str) -> tuple[str, str | None]:
    path, separator, key = value.partition("#")
    return path, key if separator else None


def load_truth(manifest_row: dict[str, Any], root: Path) -> dict[str, Any]:
    path, key = _truth_path_and_key(manifest_row["truth"])
    payload = json.loads((root / path).read_text(encoding="utf-8"))
    return payload[key] if key else payload


def load_tokens(manifest_row: dict[str, Any], root: Path) -> list[OCRToken]:
    payload = json.loads((root / manifest_row["tokens"]).read_text(encoding="utf-8"))
    rows = payload["tokens"] if isinstance(payload, dict) else payload
    return [OCRToken(row["text"], row["box"], row["confidence"]) for row in rows]


def invoice_dict(invoice: InvoiceData) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in FIELDS:
        value = getattr(invoice, field, None)
        if hasattr(value, "isoformat"):
            value = value.isoformat(timespec="seconds")
        result[field] = value
    result["items"] = [
        {"name": item.name, "quantity": item.quantity, "unit_price": item.unit_price, "amount": item.amount}
        for item in invoice.items
    ]
    return result


def _item_same(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    return (
        _same("name", expected.get("name"), predicted.get("name"))
        and _same("quantity", expected.get("quantity"), predicted.get("quantity"))
        and _same("unit_price", expected.get("unit_price"), predicted.get("unit_price"))
        and _same("amount", expected.get("amount"), predicted.get("amount"))
    )


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"samples": 0, "field_accuracy": None, "precision": None, "recall": None, "f1": None, "exact_match": None, "per_field": {}, "line_items": {}, "validation_failure_rate": None}
    per_field: dict[str, dict[str, int]] = {field: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0} for field in FIELDS}
    exact_samples = 0
    item_correct = item_total = 0
    item_exact_samples = 0
    total_tp = total_fp = total_fn = 0
    failures = 0
    for record in records:
        expected = record["truth"]
        predicted = record.get("prediction", {})
        exact = True
        for field in FIELDS:
            want, got = expected.get(field), predicted.get(field)
            matched = _same(field, want, got)
            per_field[field]["total"] += 1
            per_field[field]["correct"] += int(matched)
            want_present, got_present = _present(want), _present(got)
            per_field[field]["tp"] += int(want_present and got_present and matched)
            per_field[field]["fp"] += int(got_present and not matched)
            per_field[field]["fn"] += int(want_present and not matched)
            total_tp += int(want_present and got_present and matched)
            total_fp += int(got_present and not matched)
            total_fn += int(want_present and not matched)
            exact = exact and matched
        want_items, got_items = expected.get("items") or [], predicted.get("items") or []
        item_exact = len(want_items) == len(got_items)
        item_total += len(want_items)
        for index, want_item in enumerate(want_items):
            matched = index < len(got_items) and _item_same(want_item, got_items[index])
            item_correct += int(matched)
            item_exact = item_exact and matched
        item_exact_samples += int(item_exact)
        exact_samples += int(exact and item_exact)
        failures += int(record.get("validation_status") not in {None, "VALIDATED"})
    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    per_field_out = {}
    for field, counts in per_field.items():
        per_field_out[field] = {**counts, "accuracy": counts["correct"] / max(1, counts["total"]), **prf(counts["tp"], counts["fp"], counts["fn"])}
    overall = prf(total_tp, total_fp, total_fn)
    return {
        "samples": len(records),
        "field_accuracy": sum(counts["correct"] for counts in per_field.values()) / max(1, len(records) * len(FIELDS)),
        **overall,
        "exact_match": exact_samples / len(records),
        "per_field": per_field_out,
        "line_items": {
            "row_accuracy": item_correct / max(1, item_total),
            "exact_match": item_exact_samples / len(records),
            "expected_rows": item_total,
        },
        "validation_failure_rate": failures / len(records),
    }


def _replay_kie(row: dict[str, Any], root: Path, repeats: int) -> dict[str, Any]:
    truth = load_truth(row, root)
    tokens = load_tokens(row, root)
    extractor = KIEExtractor()
    started = time.perf_counter()
    invoice, _boxes, raw_text, details = extractor.extract_with_details(tokens)
    cold_ms = (time.perf_counter() - started) * 1000
    warm_times = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        invoice, _boxes, raw_text, details = extractor.extract_with_details(tokens)
        warm_times.append((time.perf_counter() - start) * 1000)
    confidence = sum(token.confidence for token in tokens) / max(1, len(tokens))
    validation = validate_invoice(invoice, confidence)
    predicted = invoice_dict(invoice)
    reference_text = truth.get("text")
    return {
        "fixture": row["id"], "category": row["category"], "split": row["split"],
        "truth": truth, "prediction": predicted, "validation_status": validation.status,
        "validation_issues": [issue.model_dump(mode="json") for issue in validation.issues],
        "field_details": details, "raw_text": raw_text, "confidence": confidence,
        "timings": {"cold_ms": round(cold_ms, 3), "warm_ms_median": round(statistics.median(warm_times), 3), "stage": "spatial_kie_replay"},
        "ocr_cer": cer(reference_text, raw_text) if reference_text else None,
        "ocr_wer": wer(reference_text, raw_text) if reference_text else None,
    }


def _category_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record.get("category", "unknown"), []).append(record)
    return {category: score_records(rows) for category, rows in sorted(grouped.items())}


def _latency_metrics(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    cold = [record["timings"]["cold_ms"] for record in records if record.get("timings", {}).get("cold_ms") is not None]
    warm = [record["timings"]["warm_ms_median"] for record in records if record.get("timings", {}).get("warm_ms_median") is not None]
    return {
        "cold_median_ms": round(statistics.median(cold), 3) if cold else None,
        "warm_median_ms": round(statistics.median(warm), 3) if warm else None,
        "cold_samples": len(cold),
        "warm_samples": len(warm),
        "resource_status": "unavailable: cached OCR-token replay does not measure Paddle/VLM RAM or VRAM",
    }


def run_benchmark(manifest_path: Path, root: Path, split: str = "test", strategies: tuple[str, ...] = ("ocr_kie", "vlm", "hybrid"), repeats: int = 3, live: bool = False) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = [row for row in manifest["fixtures"] if split == "all" or row["split"] == split]
    output: dict[str, Any] = {
        "schema_version": "p7-benchmark-v1", "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": manifest["dataset"], "split": split, "leakage_policy": manifest["leakage_policy"],
        "historical_baseline": manifest["historical_baseline"], "strategies": {},
    }
    for strategy in strategies:
        if strategy == "ocr_kie":
            records = [_replay_kie(row, root, repeats) for row in rows]
            metrics = score_records(records)
            metrics["ocr_cer"] = sum(r["ocr_cer"] for r in records if r["ocr_cer"] is not None) / max(1, sum(r["ocr_cer"] is not None for r in records)) if any(r["ocr_cer"] is not None for r in records) else None
            metrics["ocr_wer"] = sum(r["ocr_wer"] for r in records if r["ocr_wer"] is not None) / max(1, sum(r["ocr_wer"] is not None for r in records)) if any(r["ocr_wer"] is not None for r in records) else None
            output["strategies"][strategy] = {"status": "completed", "mode": "cached_real_ocr_tokens_replay", "metrics": metrics, "records": records, "category_metrics": _category_metrics(records), "latency": _latency_metrics(records), "unavailable_metrics": ["ocr_cer", "ocr_wer"] if metrics["ocr_cer"] is None else []}
            continue
        reason = "Chưa chạy live: manifest chỉ có OCR token dumps, ảnh gốc không còn đủ cho pipeline thị giác."
        settings = None
        if live:
            try:
                from backend.app.config import get_settings
                settings = get_settings()
                if strategy == "vlm" and settings.vlm_provider != "mock":
                    reason = "Live VLM execution requires image paths in the manifest."
                elif strategy == "hybrid" and settings.vlm_provider != "mock":
                    reason = "Live Hybrid execution requires image paths in the manifest."
                else:
                    reason = f"VLM provider hiện tại là {settings.vlm_provider!r}; không có standalone VLM/Hybrid inference để đo."
            except Exception as exc:
                reason = f"Không kiểm tra được backend live: {exc}"
        output["strategies"][strategy] = {"status": "unavailable", "reason": reason, "metrics": score_records([]), "records": []}
    return output


def write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
