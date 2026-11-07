from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_metrics import cer, entity_scores, evaluate, wer


def test_metrics_exact_and_tolerant_money(tmp_path):
    assert cer("abc", "abc") == 0
    assert wer("một hai", "một hai") == 0
    score = entity_scores({"TOTAL_COST": 93500}, {"TOTAL_COST": "93.500"})
    assert score["f1"] == 1

    fixture = tmp_path / "predictions.jsonl"
    fixture.write_text(json.dumps({"reference": {"text": "abc", "entities": {"A": "x"}}, "prediction": {"text": "abc", "entities": {"A": "x"}}}), encoding="utf-8")
    result = evaluate(fixture)
    assert result["samples"] == 1
    assert result["entity"]["f1"] == 1


def test_entity_micro_f1_aggregates_counts_over_the_whole_corpus(tmp_path):
    rows = [
        {"reference": {"entities": {"A": "x"}}, "prediction": {"entities": {"A": "x", "B": "extra"}}},
        {"reference": {"entities": {"A": "y", "B": "missing"}}, "prediction": {"entities": {"A": "y"}}},
    ]
    fixture = tmp_path / "corpus.jsonl"
    fixture.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    entity = evaluate(fixture)["entity"]
    assert entity["precision"] == 2 / 3
    assert entity["recall"] == 2 / 3
    assert entity["f1"] == 2 / 3
    assert entity["macro_f1"] == 0.5
