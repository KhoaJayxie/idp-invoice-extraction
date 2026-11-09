"""Dump real OCR tokens for KIE fixtures.

Runs the real preprocess -> PaddleOCR GPU chain on selected demo images and
stores the raw tokens (text, box, confidence) under evaluation/fixtures/
ocr_tokens/<name>.json so tests and the KIE comparator can replay them
without re-running inference.

Usage: python scripts/dump_ocr_tokens.py            # all mappings
       python scripts/dump_ocr_tokens.py vat_convert_abc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.idp_engine.ocr_detector import OCRDetector  # noqa: E402
from backend.app.idp_engine.preprocessor import ImagePreprocessor, PreprocessConfig  # noqa: E402
from backend.app.schemas.invoice import PreprocessOptions  # noqa: E402

UPLOADS = PROJECT_ROOT / "data" / "uploads"
OUT_DIR = PROJECT_ROOT / "evaluation" / "fixtures" / "ocr_tokens"

# name -> uploaded file (unique file content; duplicates by hash are skipped)
MAPPINGS = {
    "vat_blurred": "47e6f7632dcb4c18afb84b07482174e1_xuat-hoa-don-cho-cong-ty-nuoc-ngoai-7.jpg",
    "vat_convert_abc": "6704dc349763426782aa1b6d230224e2_1794117724472_4565988928609748123_4565988928609748123_36a85cce39d7ec1ed29bb9602b.jpg",
    "vat_efy_template": "13c6e453dbaa4524af7a1e941a47f3f4_1794117748473_4565988928609748123_4565988928609748123_a4959f6be0d80a60cbc224fe25.jpg",
    "vn_resto_comnieu": "a69ed8b7445f4e44bc0a2188c86c8d5f_1794118096190_4565988928609748123_4565988928609748123_6b77e1055a59c1c6cc3fdbe0b5.jpg",
    "vn_resto_gaigarden": "84ecdb8e13874cdfb4ef3403a8ee91dc_1794118136535_4565988928609748123_4565988928609748123_fd3648667fefecd6cffd91c133.jpg",
    "my_crosschannel": "fa5a3746f6d941f2b8c3b567b2eddeeb_X51005230657.jpg",
    "my_lightroom": "5e546d8ad8374aa4b2cb793f2e2a7c26_X51005303661.jpg",
}


def main() -> int:
    wanted = set(sys.argv[1:]) or set(MAPPINGS)
    settings = get_settings()
    detector = OCRDetector(
        use_paddle=settings.use_paddle,
        use_vietocr=settings.use_vietocr,
        model_dir=settings.model_dir,
        lang=settings.paddle_lang,
        device=settings.paddle_device,
        rec_batch_size=settings.paddle_rec_batch_size,
        det_limit_side_len=settings.paddle_det_limit_side_len,
        gpu_memory_fraction=settings.paddle_gpu_memory_fraction,
        strict=settings.ocr_strict,
        ocr_version=settings.paddle_ocr_version,
        model_size=settings.paddle_model_size,
    )
    preprocessor = ImagePreprocessor(
        PreprocessConfig(
            max_dimension=settings.max_image_dimension,
            max_decoded_pixels=settings.max_decoded_pixels,
        ),
        max_bytes=settings.max_upload_bytes,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, filename in MAPPINGS.items():
        if name not in wanted:
            continue
        out_path = OUT_DIR / f"{name}.json"
        if out_path.exists():
            print(f"SKIP existing {name}")
            continue
        image = UPLOADS / filename
        if not image.is_file():
            print(f"MISSING image for {name}: {filename}")
            failures += 1
            continue
        try:
            processed = preprocessor.process(image, PreprocessOptions())
            tokens = detector.detect(processed, source_path=str(image))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED {name}: {exc}")
            failures += 1
            continue
        dump = {
            "source_image": filename,
            "processed_width": processed.metadata.processed_width,
            "processed_height": processed.metadata.processed_height,
            "tokens": [
                {"text": t.text, "box": list(t.box), "confidence": round(float(t.confidence), 4)}
                for t in tokens
            ],
        }
        out_path.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK {name}: {len(tokens)} tokens -> {out_path.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
