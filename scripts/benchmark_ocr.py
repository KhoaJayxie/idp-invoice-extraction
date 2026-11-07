"""Benchmark the IDP OCR engine on real invoice images.

Usage (from the project root, venv activated):

    python scripts/benchmark_ocr.py data/uploads/*.webp --repeats 5
    python scripts/benchmark_ocr.py path/to/receipt.jpg --repeats 10 --device cpu

Reports per-image and aggregate wall-clock timings for preprocessing and
PaddleOCR inference, token counts, confidences, and GPU memory when CUDA is
active. Results are printed as a table and written to
evaluation/benchmarks/benchmark_<timestamp>.json for the project report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.idp_engine.ocr_detector import OCRDetector  # noqa: E402
from backend.app.idp_engine.pipeline import HeuristicPipeline  # noqa: E402
from backend.app.schemas.invoice import PreprocessOptions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark IDP OCR on invoice images.")
    parser.add_argument("images", nargs="+", type=Path, help="Invoice image paths.")
    parser.add_argument("--repeats", type=int, default=5, help="Warm inference repeats per image.")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default=None)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    settings = get_settings()
    device = args.device or settings.paddle_device

    print(f"PaddleOCR benchmark · device={device} · repeats={args.repeats}")
    print("=" * 78)

    # The pipeline warms the model (first call includes kernel compilation and
    # possible model downloads), so measure it separately from steady state.
    detector = OCRDetector(
        use_paddle=True,
        use_vietocr=settings.use_vietocr,
        model_dir=settings.model_dir,
        lang=settings.paddle_lang,
        device=device,
        rec_batch_size=settings.paddle_rec_batch_size,
        det_limit_side_len=settings.paddle_det_limit_side_len,
        gpu_memory_fraction=settings.paddle_gpu_memory_fraction,
        strict=False,
        ocr_version=settings.paddle_ocr_version,
        model_size=settings.paddle_model_size,
    )
    engine = detector.status()
    print(f"engine       : {engine.get('backend')} · device={engine.get('device')} · lang={engine.get('lang')}")
    print(f"models       : v={engine.get('ocr_version')} size={engine.get('model_size')}")
    if detector.backend_warning:
        print(f"warning      : {detector.backend_warning}")
    print()

    results: list[dict] = []
    for image_path in args.images:
        if not image_path.is_file():
            print(f"SKIP (not found): {image_path}")
            continue
        pipeline = HeuristicPipeline(detector=detector)
        # Cold pass: includes any one-time model/GPU warm-up.
        cold_start = time.perf_counter()
        response = pipeline.run(image_path, options=PreprocessOptions(), source_name=image_path.name)
        cold_ms = (time.perf_counter() - cold_start) * 1000

        ocr_times: list[float] = []
        preprocess_times: list[float] = []
        tokens = 0
        for _ in range(max(1, args.repeats)):
            start = time.perf_counter()
            processed = pipeline.preprocessor.process(image_path, PreprocessOptions())
            preprocess_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            detected = detector.detect(processed)
            ocr_times.append((time.perf_counter() - start) * 1000)
            tokens = max(tokens, len(detected))

        confidences = [box.confidence for box in response.boxes]
        gpu_mem = None
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                gpu_mem = round(paddle.device.cuda.memory_reserved() / 1e6, 1)
        except Exception:
            pass

        record = {
            "image": str(image_path),
            "pixels": f"{response.preprocess.processed_width}x{response.preprocess.processed_height}",
            "cold_total_ms": round(cold_ms, 1),
            "ocr_ms": {
                "median": round(statistics.median(ocr_times), 1),
                "min": round(min(ocr_times), 1),
                "max": round(max(ocr_times), 1),
            },
            "preprocess_ms_median": round(statistics.median(preprocess_times), 1),
            "tokens": tokens,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "fields": {
                "total": response.invoice.total_cost,
                "items": len(response.invoice.items),
            },
            "gpu_mem_reserved_mb": gpu_mem,
        }
        results.append(record)
        print(f"{image_path.name}")
        print(f"  size={record['pixels']}  cold_total={record['cold_total_ms']} ms")
        print(
            f"  ocr median={record['ocr_ms']['median']} ms "
            f"(min {record['ocr_ms']['min']} / max {record['ocr_ms']['max']})  "
            f"preprocess={record['preprocess_ms_median']} ms"
        )
        print(f"  tokens={record['tokens']}  avg_conf={record['avg_confidence']}  "
              f"total={record['fields']['total']}  items={record['fields']['items']}  gpu_mem={gpu_mem} MB")
        print()

    if len(results) > 1:
        ocr_medians = [r["ocr_ms"]["median"] for r in results]
        print("=" * 78)
        print(f"AGGREGATE  images={len(results)}  ocr median-of-medians="
              f"{round(statistics.median(ocr_medians), 1)} ms  "
              f"mean={round(statistics.mean(ocr_medians), 1)} ms")

    if args.json_out or results:
        out = args.json_out or (
            PROJECT_ROOT / "evaluation" / "benchmarks"
            / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device_setting": device,
            "engine": engine,
            "repeats": args.repeats,
            "results": results,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
