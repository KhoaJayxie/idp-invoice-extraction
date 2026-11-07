"""Evaluate OCR/KIE predictions from JSONL files.

Each JSONL row may contain:
{"reference": {"text": "...", "entities": {...}},
 "prediction": {"text": "...", "entities": {...}}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

try:
    from Levenshtein import distance as _levenshtein_distance
except ImportError:  # deterministic pure-Python fallback
    _levenshtein_distance = None


def normalize_text(value: Any) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: Any) -> list[str]:
    return re.findall(r"\w+", normalize_text(value), flags=re.UNICODE)


def edit_distance(left: list[Any], right: list[Any]) -> int:
    if _levenshtein_distance is not None and all(isinstance(value, str) for value in (*left, *right)):
        return int(_levenshtein_distance(left, right))
    previous = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (item != other)))
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    reference_text, prediction_text = normalize_text(reference), normalize_text(prediction)
    return edit_distance(list(reference_text), list(prediction_text)) / max(1, len(reference_text))


def wer(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = tokens(reference), tokens(prediction)
    return edit_distance(reference_tokens, prediction_tokens) / max(1, len(reference_tokens))


def canonical_entity(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    text = normalize_text(value)
    # Keep entity comparison tolerant to common money punctuation and spaces.
    if re.fullmatch(r"[-+()\d\s.,₫đvnd]+", text, flags=re.IGNORECASE):
        digits = re.sub(r"[^0-9-]", "", text)
        return digits or text
    return text


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}


def entity_scores(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    ref = {str(key): canonical_entity(value) for key, value in (reference or {}).items() if value not in (None, "")}
    pred = {str(key): canonical_entity(value) for key, value in (prediction or {}).items() if value not in (None, "")}
    keys = set(ref) | set(pred)
    per_entity: dict[str, dict[str, float | int]] = {}
    true_positive = false_positive = false_negative = 0
    for key in sorted(keys):
        matched = key in ref and key in pred and ref[key] == pred[key]
        tp = int(matched)
        fp = int(key in pred and not matched)
        fn = int(key in ref and not matched)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        per_entity[key] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}
    macro_f1 = sum(float(value["f1"]) for value in per_entity.values()) / max(1, len(per_entity))
    return {
        **_prf(true_positive, false_positive, false_negative),
        "macro_f1": macro_f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "per_entity": per_entity,
    }


def evaluate(path: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {
            "samples": 0,
            "cer": 0.0,
            "wer": 0.0,
            "entity": {**_prf(0, 0, 0), "macro_f1": 0.0, "per_entity": {}},
        }
    cer_values = []
    wer_values = []
    total_tp = total_fp = total_fn = 0
    entity_counts: dict[str, list[int]] = {}
    for row in rows:
        reference, prediction = row.get("reference", {}), row.get("prediction", {})
        cer_values.append(cer(reference.get("text", ""), prediction.get("text", "")))
        wer_values.append(wer(reference.get("text", ""), prediction.get("text", "")))
        score = entity_scores(reference.get("entities", {}), prediction.get("entities", {}))
        total_tp += int(score["tp"])
        total_fp += int(score["fp"])
        total_fn += int(score["fn"])
        for key, value in score["per_entity"].items():
            counts = entity_counts.setdefault(key, [0, 0, 0])
            counts[0] += int(value["tp"])
            counts[1] += int(value["fp"])
            counts[2] += int(value["fn"])
    per_entity = {key: {**_prf(*counts), "tp": counts[0], "fp": counts[1], "fn": counts[2]} for key, counts in sorted(entity_counts.items())}
    macro_f1 = sum(score["f1"] for score in per_entity.values()) / max(1, len(per_entity))
    return {
        "samples": len(rows),
        "cer": sum(cer_values) / len(cer_values),
        "wer": sum(wer_values) / len(wer_values),
        "entity": {**_prf(total_tp, total_fp, total_fn), "macro_f1": macro_f1, "per_entity": per_entity},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính CER, WER và Entity Precision/Recall/F1 từ JSONL.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.jsonl), ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
