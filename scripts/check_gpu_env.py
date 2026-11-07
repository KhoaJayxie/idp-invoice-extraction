"""Verify the Python / Paddle / CUDA / NVIDIA driver chain for IDP V2.

Run before the first real-mode OCR use (from the project root, venv active):

    python scripts/check_gpu_env.py

Exit code 0 means everything needed for GPU OCR is present; 1 means the GPU
path is unavailable (the app will fall back to CPU) or the environment is
broken; 2 means Python-level problems (paddle not importable).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(label: str, ok: bool, detail: str, warn_only: bool = False) -> bool:
    status = PASS if ok else (WARN if warn_only else FAIL)
    print(f"[{status}] {label}: {detail}")
    return ok or warn_only


def main() -> int:
    print("IDP V2 environment check")
    print("=" * 78)

    check(
        "Python",
        sys.version_info >= (3, 10),
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"({'OK (>=3.10)' if sys.version_info >= (3, 10) else 'too old'})",
    )

    nvidia_smi = shutil.which("nvidia-smi")
    driver_line = ""
    if nvidia_smi:
        try:
            output = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            driver_line = output.splitlines()[0] if output else ""
            check("NVIDIA driver", bool(driver_line), driver_line)
        except Exception as exc:  # pragma: no cover
            check("NVIDIA driver", False, f"nvidia-smi failed: {exc}", warn_only=True)
    else:
        check("NVIDIA driver", False, "nvidia-smi not found — GPU OCR unavailable", warn_only=True)

    try:
        import paddle
    except Exception as exc:
        print(f"[{FAIL}] PaddlePaddle: import failed ({exc})")
        return 2

    compiled = paddle.device.is_compiled_with_cuda()

    def _version_of(value) -> str:
        value = getattr(paddle.version, value, "?")
        return value() if callable(value) else str(value)

    check(
        "PaddlePaddle build",
        True,
        f"{paddle.__version__} (CUDA compiled: {compiled}, "
        f"CUDA {_version_of('cuda')}, cuDNN {_version_of('cudnn')})",
    )

    gpu_ok = False
    if compiled:
        try:
            paddle.set_device("gpu")
            probe = paddle.randn([8, 8], dtype="float32")
            _ = probe @ probe
            paddle.device.synchronize()
            capability = paddle.device.cuda.get_device_capability(0)
            gpu_ok = True
            check(
                "GPU inference",
                True,
                f"{paddle.device.get_device()} · compute capability "
                f"{capability[0]}.{capability[1]} · matmul OK",
            )
            try:
                paddle.utils.run_check()
                check("paddle run_check", True, "GPU runtime verified")
            except Exception as exc:
                check("paddle run_check", False, f"failed: {exc}", warn_only=True)
        except Exception as exc:
            check("GPU inference", False, f"GPU init failed ({exc}); OCR will use CPU", warn_only=True)
    else:
        check("GPU inference", False, "paddle build has no CUDA support; OCR will use CPU", warn_only=True)

    try:
        import paddleocr  # noqa: F401

        check("PaddleOCR", True, f"installed ({getattr(paddleocr, '__version__', 'unknown')})")
    except Exception as exc:
        print(f"[{FAIL}] PaddleOCR: not importable ({exc})")
        return 2

    models_dir = Path.home() / ".paddlex" / "official_models"
    cached = sorted(path.name for path in models_dir.iterdir()) if models_dir.is_dir() else []
    check(
        "Model cache",
        True,
        f"{len(cached)} cached in {models_dir}" + (f": {', '.join(cached)}" if cached else " (first run will download)"),
    )

    print("=" * 78)
    if gpu_ok:
        print("Result: READY — real-mode OCR will run on the GPU.")
        return 0
    print("Result: CPU FALLBACK — real-mode OCR works but will run on the CPU.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
