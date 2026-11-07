"""Run the full IDP pipeline over an image folder and score it against JSONL truth.

This is the project-report evaluation entry point: it executes the real
preprocess -> OCR -> KIE -> validate chain (not stored predictions), writes
per-sample predictions to JSONL, and then measures CER/WER and entity
precision/recall/F1 with ``evaluation.evaluate_metrics``.

Ground-truth format (one JSON object per line):

    {"image": "data/uploads/receipt.webp",
     "reference": {"text": "...", "entities": {"SELLER": "...", "TOTAL_COST": 12000000}}}

Usage:

    python scripts/evaluate_pipeline.py evaluation/fixtures/invoice_ground_truth.jsonl
    python scripts/evaluate_pipeline.py truth.jsonl --mode vlm --out evaluation/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.idp_engine.pipeline import HeuristicPipeline, VLMPipeline  # noqa: E402
from backend.app.schemas.invoice import PreprocessOptions  # noqa: E402
from evaluation.evaluate_metrics import canonical_entity, evaluate  # noqa: E402

ENTITY_KEYS = {
    "SELLER": "seller",
    "ADDRESS": "address",
    "INVOICE_ID": "invoice_id",
    "TIMESTAMP": "timestamp",
    "SUBTOTAL": "subtotal",
    "TAX": "tax",
    "DISCOUNT": "discount",
    "TOTAL_COST": "total_cost",
}


def prediction_entities(invoice) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    for entity_key, field in ENTITY_KEYS.items():
        value = getattr(invoice, field, None)
        if value in (None, "", 0, 0.0):
            continue
        entities[entity_key] = value.date().isoformat() if isinstance(value, datetime) else value
    return entities


def main() -> int:
    parser = argparse.ArgumentParser(description="IDP end-to-end evaluation on real images.")
    parser.add_argument("dataset", type=Path, help="JSONL with image paths and references.")
    parser.add_argument("--mode", choices=["heuristic", "vlm"], default="heuristic")
    parser.add_argument("--out", type=Path, default=None, help="Where to write predictions JSONL.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Base dir for relative image paths.")
    args = parser.parse_args()

    settings = get_settings()
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print("Dataset is empty.")
        return 1

    pipeline = HeuristicPipeline() if args.mode == "heuristic" else VLMPipeline(
        settings.vlm_provider, settings.openai_api_key or settings.gemini_api_key, settings.vlm_model
    )
    out_path = args.out or (
        PROJECT_ROOT / "evaluation" / "predictions"
        / f"predictions_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_value = row.get("image")
            reference = row.get("reference", {})
            if not image_value:
                # A row without an image is a stored-prediction row; keep it as-is.
                scored.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = args.root / image_path
            if not image_path.is_file():
                print(f"SKIP missing image: {image_path}")
                continue
            try:
                response = pipeline.run(image_path, options=PreprocessOptions(), source_name=image_path.name)
            except Exception as exc:
                print(f"FAILED {image_path.name}: {exc}")
                continue
            prediction = {
                "text": response.raw_text,
                "entities": prediction_entities(response.invoice),
            }
            entry = {
                "image": str(image_path),
                "reference": reference,
                "prediction": prediction,
                "status": response.status,
                "confidence": round(response.confidence, 4),
                "timings": response.timings,
                "engine": response.engine,
            }
            scored.append(entry)
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            matched = sum(
                1 for key, value in prediction["entities"].items()
                if canonical_entity(reference.get("entities", {}).get(key))
                == canonical_entity(value)
            )
            print(
                f"{image_path.name}: status={response.status} conf={response.confidence:.3f} "
                f"entities={len(prediction['entities'])} matched={matched} "
                f"ocr={response.timings.get('ocr_ms')}ms"
            )

    metrics = evaluate(out_path)
    print("\n" + "=" * 78)
    print(f"samples : {metrics['samples']}")
    print(f"CER     : {metrics['cer']:.4f}")
    print(f"WER     : {metrics['wer']:.4f}")
    entity = metrics["entity"]
    print(
        f"Entity  : P={entity['precision']:.4f} R={entity['recall']:.4f} "
        f"F1={entity['f1']:.4f} macroF1={entity['macro_f1']:.4f}"
    )
    print(f"predictions: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
