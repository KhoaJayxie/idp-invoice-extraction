"""Run the P7 benchmark and write a JSON result consumed by the dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark_runner import run_benchmark, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy benchmark P7 có split và provenance.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation" / "benchmark_manifest.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strategies", nargs="+", choices=["ocr_kie", "vlm", "hybrid"], default=["ocr_kie", "vlm", "hybrid"])
    parser.add_argument("--live", action="store_true", help="Chỉ bật khi manifest có ảnh gốc và backend VLM thật.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = run_benchmark(args.manifest, ROOT, args.split, tuple(args.strategies), args.repeats, args.live)
    out = (args.out if args.out.is_absolute() else ROOT / args.out) if args.out else ROOT / "evaluation" / "benchmarks" / f"benchmark_{args.split}.json"
    write_json(result, out)
    print(f"saved: {out.relative_to(ROOT)}")
    for name, strategy in result["strategies"].items():
        print(f"{name}: {strategy['status']} - {strategy.get('metrics', {}).get('samples', 0)} samples")
        if strategy["status"] == "unavailable":
            print(f"  reason: {strategy['reason'].encode('ascii', 'replace').decode('ascii')}")
        else:
            metrics = strategy["metrics"]
            print(f"  field_accuracy={metrics['field_accuracy']:.4f} f1={metrics['f1']:.4f} exact={metrics['exact_match']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
