from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from ..config import get_settings
from ..schemas.invoice import (
    Box,
    ExtractionResponse,
    InvoiceData,
    PreprocessOptions,
)
from .kie_extractor import KIEExtractor
from .ocr_detector import OCRDetector
from .postprocessor import normalize_datetime, normalize_money, normalize_text, validate_invoice
from .preprocessor import ImagePreprocessor, PreprocessConfig, PreprocessResult
from .vlm_extractor import VLMExtractor


class IDPPipeline(ABC):
    strategy: str

    @abstractmethod
    def run(
        self,
        image_path: str | Path,
        options: PreprocessOptions | dict[str, Any] | None = None,
        source_name: str | None = None,
    ) -> ExtractionResponse:
        raise NotImplementedError


class HeuristicPipeline(IDPPipeline):
    strategy = "2-stage-paddleocr-vietocr-kie"

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        detector: OCRDetector | None = None,
        kie: KIEExtractor | None = None,
        artifact_dir: str | Path | None = None,
    ):
        settings = get_settings()
        self.preprocessor = preprocessor or ImagePreprocessor(
            PreprocessConfig(
                max_dimension=settings.max_image_dimension,
                max_decoded_pixels=settings.max_decoded_pixels,
            ),
            max_bytes=settings.max_upload_bytes,
        )
        self.detector = detector or OCRDetector(
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
        self.kie = kie or KIEExtractor()
        self.artifact_dir = Path(artifact_dir or settings.processed_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_path: str | Path,
        options: PreprocessOptions | dict[str, Any] | None = None,
        source_name: str | None = None,
    ) -> ExtractionResponse:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        step = time.perf_counter()
        processed = self.preprocessor.process(image_path, options)
        timings["preprocess_ms"] = round((time.perf_counter() - step) * 1000, 1)
        step = time.perf_counter()
        tokens = self.detector.detect(processed, source_path=image_path)
        timings["ocr_ms"] = round((time.perf_counter() - step) * 1000, 1)
        step = time.perf_counter()
        data, raw_boxes, raw_text = self.kie.extract(tokens)
        timings["kie_ms"] = round((time.perf_counter() - step) * 1000, 1)
        # Overall confidence is the mean token confidence: a single degraded
        # token (one misread character) should not sink an otherwise clean
        # extraction, while consistently weak recognition still will.
        if tokens:
            confidence = sum(token.confidence for token in tokens) / len(tokens)
        else:
            confidence = 0.0
        data = _normalize_invoice(data)
        settings = get_settings()
        validation = validate_invoice(data, confidence, settings.review_confidence_threshold)
        artifact = self._write_artifacts(processed, source_name or Path(image_path).stem)
        boxes = [Box.model_validate(box) for box in raw_boxes]
        warnings = list(processed.metadata.warnings) + validation.warnings
        if self.detector.backend_warning and not settings.ocr_strict:
            warnings.append(self.detector.backend_warning)
        if not tokens:
            warnings.append("Không có kết quả OCR; hãy cài model OCR hoặc cung cấp văn bản OCR demo.")
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return ExtractionResponse(
            invoice=data,
            status=validation.status,
            confidence=confidence,
            boxes=boxes,
            warnings=warnings,
            strategy=self.strategy,
            raw_text=raw_text,
            math=validation.math,
            preprocess=processed.metadata,
            source_image=source_name,
            original_filename=Path(image_path).name,
            aligned_image_url=f"/processed/{artifact['aligned']}",
            processed_image_url=f"/processed/{artifact['enhanced']}",
            binary_image_url=f"/processed/{artifact['binary']}",
            timings=timings,
            engine=self.detector.status(),
        )

    def _write_artifacts(self, processed: PreprocessResult, stem: str) -> dict[str, str]:
        safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)[:60] or uuid4().hex
        token = uuid4().hex[:10]
        enhanced_name = f"{safe_stem}_{token}_enhanced.png"
        binary_name = f"{safe_stem}_{token}_binary.png"
        aligned_name = f"{safe_stem}_{token}_aligned.png"
        if not cv2.imwrite(str(self.artifact_dir / enhanced_name), processed.enhanced_gray):
            raise RuntimeError("Không thể lưu ảnh enhanced cho OCR/review.")
        if not cv2.imwrite(str(self.artifact_dir / binary_name), processed.binary):
            raise RuntimeError("Không thể lưu ảnh binary cho review.")
        if not cv2.imwrite(str(self.artifact_dir / aligned_name), processed.original_bgr):
            raise RuntimeError("Không thể lưu ảnh căn chỉnh cho review.")
        return {"enhanced": enhanced_name, "binary": binary_name, "aligned": aligned_name}


class VLMPipeline(IDPPipeline):
    strategy = "vlm-structured-json"

    def __init__(
        self,
        provider: str = "mock",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        preprocessor: ImagePreprocessor | None = None,
        vlm: VLMExtractor | None = None,
        artifact_dir: str | Path | None = None,
    ):
        settings = get_settings()
        self.preprocessor = preprocessor or ImagePreprocessor(
            PreprocessConfig(
                max_dimension=settings.max_image_dimension,
                max_decoded_pixels=settings.max_decoded_pixels,
            ),
            max_bytes=settings.max_upload_bytes,
        )
        self.vlm = vlm or VLMExtractor(provider, api_key, model)
        self.artifact_dir = Path(artifact_dir or settings.processed_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_path: str | Path,
        options: PreprocessOptions | dict[str, Any] | None = None,
        source_name: str | None = None,
    ) -> ExtractionResponse:
        processed = self.preprocessor.process(image_path, options)
        artifact = self._write_artifacts(processed, source_name or Path(image_path).stem)
        vlm_image = self.artifact_dir / artifact["aligned"]
        data, raw_boxes, raw_text = self.vlm.extract(vlm_image)
        data = _normalize_invoice(data)
        settings = get_settings()
        validation = validate_invoice(data, 0.90, settings.review_confidence_threshold)
        return ExtractionResponse(
            invoice=data,
            status=validation.status,
            confidence=0.90,
            boxes=[Box.model_validate(box) for box in raw_boxes],
            warnings=list(processed.metadata.warnings) + validation.warnings,
            strategy=self.strategy,
            raw_text=raw_text,
            math=validation.math,
            preprocess=processed.metadata,
            source_image=source_name,
            original_filename=Path(image_path).name,
            aligned_image_url=f"/processed/{artifact['aligned']}",
            processed_image_url=f"/processed/{artifact['enhanced']}",
            binary_image_url=f"/processed/{artifact['binary']}",
        )

    def _write_artifacts(self, processed: PreprocessResult, stem: str) -> dict[str, str]:
        safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)[:60] or uuid4().hex
        token = uuid4().hex[:10]
        enhanced_name = f"{safe_stem}_{token}_enhanced.png"
        binary_name = f"{safe_stem}_{token}_binary.png"
        aligned_name = f"{safe_stem}_{token}_aligned.png"
        cv2.imwrite(str(self.artifact_dir / enhanced_name), processed.enhanced_gray)
        cv2.imwrite(str(self.artifact_dir / binary_name), processed.binary)
        cv2.imwrite(str(self.artifact_dir / aligned_name), processed.original_bgr)
        return {"enhanced": enhanced_name, "binary": binary_name, "aligned": aligned_name}


def _normalize_invoice(data: InvoiceData) -> InvoiceData:
    payload = data.model_dump()
    for key in ("seller", "address", "invoice_id"):
        if payload.get(key):
            payload[key] = normalize_text(str(payload[key]))
    payload["timestamp"] = normalize_datetime(payload.get("timestamp"))
    for key in ("subtotal", "tax", "discount", "total_cost"):
        payload[key] = normalize_money(payload.get(key))
    normalized_items = []
    for item in payload.get("items", []):
        item["name"] = normalize_text(item.get("name", ""))
        item["quantity"] = normalize_money(item.get("quantity", 1))
        item["unit_price"] = normalize_money(item.get("unit_price", 0))
        item["amount"] = normalize_money(item.get("amount", 0)) or item["quantity"] * item["unit_price"]
        normalized_items.append(item)
    payload["items"] = normalized_items
    if payload["subtotal"] <= 0 and normalized_items:
        payload["subtotal"] = sum(item["amount"] for item in normalized_items)
    return InvoiceData.model_validate(payload)
