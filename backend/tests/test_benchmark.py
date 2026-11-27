from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark_runner import load_manifest, run_benchmark, score_records


def _record(expected, predicted, status="VALIDATED"):
    return {
        "truth": expected,
        "prediction": predicted,
        "validation_status": status,
    }


def test_score_records_uses_numeric_tolerance_and_reports_items():
    record = _record(
        {
            "seller": "Cửa hàng A",
            "subtotal": 100.0,
            "items": [{"name": "Bút", "quantity": 2, "unit_price": 50, "amount": 100}],
        },
        {
            "seller": " cửa hàng a ",
            "subtotal": 100.04,
            "items": [{"name": "Bút", "quantity": 2.0, "unit_price": 50, "amount": 100}],
        },
    )

    metrics = score_records([record])

    assert metrics["samples"] == 1
    assert metrics["per_field"]["seller"]["accuracy"] == 1.0
    assert metrics["per_field"]["subtotal"]["accuracy"] == 1.0
    assert metrics["line_items"] == {"row_accuracy": 1.0, "exact_match": 1.0, "expected_rows": 1}
    assert metrics["exact_match"] == 1.0
    assert metrics["validation_failure_rate"] == 0.0


def test_score_records_distinguishes_missing_fields_from_false_positives():
    metrics = score_records(
        [
            _record(
                {"seller": "A", "buyer": None, "items": []},
                {"seller": "B", "buyer": "Unexpected", "items": []},
                status="NEEDS_REVIEW",
            )
        ]
    )

    seller = metrics["per_field"]["seller"]
    buyer = metrics["per_field"]["buyer"]
    assert seller["tp"] == 0 and seller["fp"] == 1 and seller["fn"] == 1
    assert buyer["tp"] == 0 and buyer["fp"] == 1 and buyer["fn"] == 0
    assert metrics["validation_failure_rate"] == 1.0


def test_run_benchmark_respects_manifest_split_and_marks_unavailable_strategies():
    result = run_benchmark(
        ROOT / "evaluation" / "benchmark_manifest.json",
        ROOT,
        split="dev",
        strategies=("vlm", "hybrid"),
    )

    assert result["split"] == "dev"
    assert result["strategies"]["vlm"]["status"] == "unavailable"
    assert result["strategies"]["hybrid"]["status"] == "unavailable"
    assert "ảnh gốc" in result["strategies"]["vlm"]["reason"]
    assert result["strategies"]["vlm"]["metrics"]["samples"] == 0


def test_manifest_has_frozen_dev_and_test_splits():
    manifest = load_manifest(ROOT / "evaluation" / "benchmark_manifest.json")
    splits = {row["split"] for row in manifest["fixtures"]}
    assert splits == {"dev", "test"}
    assert sum(row["split"] == "dev" for row in manifest["fixtures"]) == 4
    assert sum(row["split"] == "test" for row in manifest["fixtures"]) == 7
