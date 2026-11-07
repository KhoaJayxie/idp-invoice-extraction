from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from backend.app.idp_engine.preprocessor import ImagePreprocessingError, ImagePreprocessor, PreprocessConfig


def synthetic_receipt() -> np.ndarray:
    canvas = np.full((700, 900, 3), 115, dtype=np.uint8)
    polygon = np.array([[170, 55], [755, 90], [710, 645], [125, 600]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, polygon, (245, 245, 245))
    cv2.polylines(canvas, [polygon], True, (250, 250, 250), 5)
    cv2.putText(canvas, "HOA DON VIET NAM", (240, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (30, 30, 30), 2)
    cv2.putText(canvas, "Tong cong: 93.500", (230, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return canvas


def test_full_preprocessing_detects_page_and_returns_variants():
    processor = ImagePreprocessor(PreprocessConfig(max_dimension=800))
    result = processor.process(synthetic_receipt())

    assert result.enhanced_gray.ndim == 2
    assert result.binary.shape == result.enhanced_gray.shape
    assert result.metadata.original_width == 900
    assert result.metadata.original_height == 700
    assert result.metadata.processed_width <= 800
    assert result.metadata.processed_height <= 800
    assert result.metadata.binarization_applied is True
    assert result.metadata.blur_score >= 0
    assert result.metadata.document_detected is True
    assert result.metadata.perspective_applied is True


def test_preprocessing_options_can_disable_steps():
    processor = ImagePreprocessor(PreprocessConfig(max_dimension=500))
    result = processor.process(
        synthetic_receipt(),
        {"auto_perspective": False, "deskew": False, "binarize": False},
    )

    assert result.metadata.perspective_applied is False
    assert result.metadata.deskew_applied is False
    assert result.metadata.binarization_applied is False
    assert not np.any(result.binary)
    assert any("tắt" in warning for warning in result.metadata.warnings)


def test_bytes_decode_applies_exif_and_rejects_invalid_input():
    image = Image.new("RGB", (120, 80), (255, 255, 255))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90° clockwise
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)
    result = ImagePreprocessor(PreprocessConfig(auto_perspective=False, deskew=False, max_dimension=200)).process(output.getvalue())
    assert result.metadata.original_width == 80
    assert result.metadata.original_height == 120

    with pytest.raises(ImagePreprocessingError):
        ImagePreprocessor().process(b"not an image")


def test_decoded_pixel_limit_and_deskew_are_applied():
    text = np.full((480, 700), 255, dtype=np.uint8)
    for y in (130, 210, 290, 370):
        cv2.putText(text, "INVOICE 12345 TOTAL 93500", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((350, 240), 5.0, 1.0)
    rotated = cv2.warpAffine(text, matrix, (700, 480), borderValue=255)
    result = ImagePreprocessor(PreprocessConfig(auto_perspective=False, illumination_correction=False, denoise=False, clahe=False, sharpen=False, binarize=False)).process(rotated)
    assert result.metadata.deskew_applied is True
    assert abs(result.metadata.deskew_angle) >= 2

    with pytest.raises(ImagePreprocessingError, match="quá nhiều điểm ảnh"):
        ImagePreprocessor(PreprocessConfig(max_decoded_pixels=1000)).process(np.zeros((50, 50), dtype=np.uint8))
