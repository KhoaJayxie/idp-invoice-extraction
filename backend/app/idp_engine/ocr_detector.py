from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from .preprocessor import PreprocessResult


class OCRUnavailableError(RuntimeError):
    """Raised in strict real mode when the configured OCR engine cannot run.

    Routes translate this into HTTP 503 so the API never fabricates results
    with sidecar/demo text while a real model was requested.
    """


@dataclass(slots=True)
class OCRToken:
    text: str
    box: list[float]
    confidence: float
    source: str = "paddleocr"


class OCRDetector:
    """PaddleOCR/VietOCR adapter with a deterministic sidecar fallback.

    PaddleOCR and VietOCR are intentionally imported lazily. A base installation
    can therefore demonstrate preprocessing, KIE and validation without model
    downloads, while production can opt in to the real recognizers.

    Real mode (`use_paddle=True`) never reads the sidecar `.txt` fallback:
    when `strict=True` (the default) a missing/unloading engine raises
    OCRUnavailableError instead of pretending to extract. Set `strict=False`
    only for the offline demo flow.
    """

    def __init__(
        self,
        use_paddle: bool = False,
        use_vietocr: bool = False,
        model_dir: str | Path | None = None,
        *,
        lang: str = "vi",
        device: str = "auto",
        rec_batch_size: int = 16,
        det_limit_side_len: int = 960,
        gpu_memory_fraction: float = 0.5,
        strict: bool = True,
        ocr_version: str = "PP-OCRv6",
        model_size: str = "medium",
    ):
        self.use_paddle = use_paddle
        self.use_vietocr = use_vietocr
        self.model_dir = Path(model_dir) if model_dir else None
        self.lang = lang or "vi"
        self.device_preference = (device or "auto").lower()
        self.rec_batch_size = max(1, int(rec_batch_size))
        self.det_limit_side_len = max(256, int(det_limit_side_len))
        fraction = float(gpu_memory_fraction or 0.5)
        self.gpu_memory_fraction = fraction if 0 < fraction < 1 else 0.5
        self.strict = bool(strict)
        self.ocr_version = ocr_version or "PP-OCRv6"
        self.model_size = (model_size or "medium").lower()
        self.ocr: Any = None
        self.vietocr_predictor: Any = None
        self.backend_warning: str | None = None
        self.device_used: str = "n/a"
        self._predict_lock = Lock()
        self.engine_info: dict[str, Any] = {
            "backend": "sidecar",
            "device": "n/a",
            "lang": self.lang,
            "strict": self.strict,
        }
        if use_paddle:
            self._load_paddle()
        if use_vietocr:
            self._load_vietocr()

    # ------------------------------------------------------------------ load

    def _load_paddle(self) -> None:
        try:
            import paddle
        except Exception as exc:  # pragma: no cover - exercised only without paddle
            self.backend_warning = f"PaddlePaddle chưa được cài trong môi trường này: {exc}"
            if self.strict:
                raise OCRUnavailableError(self.backend_warning) from exc
            return

        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:
            self.backend_warning = f"PaddleOCR chưa được cài trong môi trường này: {exc}"
            if self.strict:
                raise OCRUnavailableError(self.backend_warning) from exc
            return

        wants_gpu = self.device_preference in {"gpu", "auto"}
        compiled_cuda = bool(paddle.device.is_compiled_with_cuda()) if hasattr(paddle, "device") else False
        device: str | None = None
        gpu_error: str | None = None
        if wants_gpu and compiled_cuda:
            try:
                paddle.set_flags({"FLAGS_fraction_of_gpu_memory_to_use": self.gpu_memory_fraction})
            except Exception:
                pass
            try:
                paddle.set_device("gpu")
                probe = paddle.randn([4, 4], dtype="float32")
                _ = probe @ probe
                paddle.device.synchronize()
                device = "gpu"
            except Exception as exc:  # GPU present but unusable (driver/runtime mismatch)
                gpu_error = str(exc)
                try:
                    paddle.set_device("cpu")
                except Exception:
                    pass
        if device is None and self.device_preference == "gpu":
            # Forced GPU but unusable — fail fast in real mode.
            self.backend_warning = (
                f"Yêu cầu GPU nhưng không khởi tạo được: {gpu_error}"
            )
            if self.strict:
                raise OCRUnavailableError(self.backend_warning)
            device = "cpu"
        if device is None:
            device = "cpu"
            try:
                paddle.set_device("cpu")
            except Exception:
                pass

        self.device_used = device
        self.engine_info = {
            "backend": "paddleocr",
            "device": device,
            "lang": self.lang,
            "ocr_version": self.ocr_version,
            "model_size": self.model_size,
            "strict": self.strict,
            "rec_batch_size": self.rec_batch_size,
            "det_limit_side_len": self.det_limit_side_len,
            "gpu_memory_fraction": self.gpu_memory_fraction if device == "gpu" else None,
        }
        if gpu_error and device == "cpu" and self.device_preference == "auto":
            self.backend_warning = f"GPU không khả dụng, đã fallback sang CPU: {gpu_error}"

        try:
            self.ocr = self._build_paddle_ocr(PaddleOCR, device)
        except Exception as exc:
            self.backend_warning = f"PaddleOCR chưa sẵn sàng: {exc}"
            if self.strict:
                raise OCRUnavailableError(self.backend_warning) from exc
            self.ocr = None

    # PaddleOCR 3.x forwards these through **kwargs into the PaddleX
    # predictor (they are not named constructor parameters), so they must
    # survive signature filtering.
    _PASSTHROUGH_KWARGS = frozenset({"device", "precision", "enable_mkldnn", "cpu_threads", "use_hpip", "use_tensorrt"})

    def _build_paddle_ocr(self, PaddleOCR: Any, device: str) -> Any:
        """Instantiate PaddleOCR 3.x while staying compatible with 2.x."""
        wants: dict[str, Any] = {
            "lang": self.lang,
            "ocr_version": self.ocr_version,
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "text_det_limit_side_len": self.det_limit_side_len,
            "text_det_limit_type": "max",
            "text_recognition_batch_size": self.rec_batch_size,
        }
        # Pin the model size within the family so demo runs are reproducible;
        # None lets PaddleOCR pick its default. Size vocabularies differ per
        # version: PP-OCRv5 uses mobile|server, PP-OCRv6 uses tiny|small|medium.
        size_vocab = {
            "PP-OCRv5": ("mobile", "server"),
            "PP-OCRv6": ("tiny", "small", "medium"),
        }.get(self.ocr_version.upper())
        size = self.model_size if size_vocab and self.model_size in size_vocab else None
        wants["text_detection_model_name"] = f"{self.ocr_version}_{size}_det" if size else None
        wants["text_recognition_model_name"] = f"{self.ocr_version}_{size}_rec" if size else None
        if device == "gpu":
            wants["device"] = "gpu:0"
        else:
            wants["device"] = "cpu"
            wants["cpu_threads"] = max(1, os.cpu_count() or 4)
            # Paddle 3.4 OneDNN execution crashes on PP-OCR PIR models
            # (ConvertPirAttribute2RuntimeError), so plain CPU kernels are
            # used for the fallback path.
            wants["enable_mkldnn"] = False
        try:
            params = inspect.signature(PaddleOCR.__init__).parameters
            kwargs = {
                key: value
                for key, value in wants.items()
                if key in params or key in self._PASSTHROUGH_KWARGS
            }
        except (TypeError, ValueError):
            kwargs = dict(wants)
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            # PaddleOCR 2.x constructor shape.
            legacy = {
                "use_angle_cls": True,
                "lang": self.lang,
                "show_log": False,
                "det_limit_side_len": self.det_limit_side_len,
                "det_limit_type": "max",
                "rec_batch_num": self.rec_batch_size,
            }
            return PaddleOCR(**legacy)

    def _load_vietocr(self) -> None:
        try:
            from vietocr.tool.config import Cfg  # type: ignore
            from vietocr.tool.predictor import Predictor  # type: ignore

            config = Cfg.load_config_from_name("vgg_transformer")
            if self.model_dir:
                # VietOCR accepts local weights paths through this config key;
                # leave the default when a project model is not present.
                candidate = self.model_dir / "vietocr.pth"
                if candidate.exists():
                    config["weights"] = str(candidate)
            config["device"] = "cpu"
            self.vietocr_predictor = Predictor(config)
        except Exception as exc:
            self.backend_warning = f"VietOCR chưa sẵn sàng: {exc}"
            self.vietocr_predictor = None

    def status(self) -> dict[str, Any]:
        """Serializable engine state for /api/ocr/status and Extract responses."""
        return dict(self.engine_info)

    # --------------------------------------------------------------- detect

    def detect(
        self,
        image: PreprocessResult | Any,
        source_path: str | Path | None = None,
    ) -> list[OCRToken]:
        processed_image = image.ocr_image if isinstance(image, PreprocessResult) else image
        if self.ocr is not None:
            tokens = self._detect_with_paddle(processed_image)
            if tokens:
                if self.vietocr_predictor is not None and isinstance(image, PreprocessResult):
                    tokens = self._replace_text_with_vietocr(tokens, image.ocr_image)
                return tokens
            # Paddle ran and found nothing — that is a real (empty) result.
            # Only the non-strict demo flow may fall back to sidecar text.
            if self.strict or source_path is None:
                return []
        elif self.use_paddle and self.strict:
            # Real mode was requested but the engine never came up.
            raise OCRUnavailableError(
                self.backend_warning or "PaddleOCR chưa sẵn sàng."
            )
        if source_path is not None:
            height, width = processed_image.shape[:2]
            fallback = self._read_sidecar(Path(source_path), width=width, height=height)
            if fallback:
                return fallback
        return []

    def _detect_with_paddle(self, image: Any) -> list[OCRToken]:
        if image is None:
            return []
        engine_image = image
        try:
            import numpy as np

            if isinstance(image, np.ndarray) and image.ndim == 2:
                engine_image = np.repeat(image[:, :, None], 3, axis=2)
        except Exception:
            pass
        with self._predict_lock:
            try:
                result = self.ocr.predict(engine_image)
            except Exception as exc:
                # Never swallow inference failures silently: record them so
                # the API can explain a degraded result, and fail loudly in
                # strict real mode instead of returning an empty invoice.
                self.backend_warning = f"Lỗi suy luận PaddleOCR: {exc}"
                if self.strict:
                    raise OCRUnavailableError(self.backend_warning) from exc
                try:
                    result = self.ocr.ocr(engine_image, cls=True)
                except Exception:
                    return []
        return self._parse_paddle_result(result, source_shape=getattr(engine_image, "shape", None))

    def _parse_paddle_result(self, result: Any, source_shape: tuple[int, ...] | None = None) -> list[OCRToken]:
        """Normalize PaddleOCR 2.x lists and 3.x result objects."""
        if result is None:
            return []
        if hasattr(result, "json"):
            try:
                result = result.json
                if callable(result):
                    result = result()
            except Exception:
                pass
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return []
        if isinstance(result, dict):
            return self._parse_paddle_dict(result, source_shape)
        if hasattr(result, "to_dict"):
            try:
                return self._parse_paddle_dict(result.to_dict(), source_shape)
            except Exception:
                pass
        if isinstance(result, (list, tuple)):
            # PaddleOCR 2.x uses either one line or a list of lines wrapped in
            # one list per image. Detect the line shape before unwrapping so a
            # single-line receipt does not get mistaken for a point list.
            if self._is_legacy_line(result):
                return self._parse_legacy_lines([result])
            if len(result) == 1 and isinstance(result[0], (list, tuple)):
                nested = self._parse_paddle_result(result[0], source_shape)
                if nested:
                    return nested
            tokens: list[OCRToken] = []
            for line in result:
                if isinstance(line, dict) or hasattr(line, "json") or hasattr(line, "to_dict"):
                    tokens.extend(self._parse_paddle_result(line, source_shape))
                elif self._is_legacy_line(line):
                    tokens.extend(self._parse_legacy_lines([line]))
            return sorted(tokens, key=lambda token: (token.box[1], token.box[0]))
        return []

    def _parse_legacy_lines(self, lines: list[Any]) -> list[OCRToken]:
        tokens: list[OCRToken] = []
        for line in lines:
            points, recognition = line[0], line[1]
            text, confidence = recognition[0], recognition[1]
            box = self._box_from_points(points)
            if box is not None and str(text).strip():
                tokens.append(OCRToken(str(text).strip(), box, self._score(confidence)))
        return sorted(tokens, key=lambda token: (token.box[1], token.box[0]))

    @staticmethod
    def _is_legacy_line(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        recognition = value[1]
        return (
            isinstance(recognition, (list, tuple))
            and len(recognition) >= 2
            and isinstance(recognition[0], (str, bytes))
        )

    def _parse_paddle_dict(self, payload: dict[str, Any], source_shape: tuple[int, ...] | None = None) -> list[OCRToken]:
        # Paddle 3.x JSON commonly stores rec_texts/rec_scores/dt_polys arrays.
        texts = self._first_value(payload, "rec_texts", "texts", "text")
        scores = self._first_value(payload, "rec_scores", "scores")
        boxes = self._first_value(payload, "dt_polys", "rec_polys", "polys", "boxes")
        text_values = self._as_sequence(texts)
        if text_values is None:
            for value in payload.values():
                if isinstance(value, dict):
                    nested = self._parse_paddle_dict(value, source_shape)
                    if nested:
                        return nested
            return []
        score_values = self._as_sequence(scores) or []
        box_values = self._as_sequence(boxes) or []
        # Placeholder boxes must live in the real coordinate space of the
        # image sent to the engine, not a hardcoded 800px canvas.
        placeholder_width = float(source_shape[1]) if source_shape and len(source_shape) >= 2 else 800.0
        placeholder_height = float(source_shape[0]) if source_shape and len(source_shape) >= 1 else 2000.0
        row_pitch = max(12.0, placeholder_height / max(1, len(text_values)))
        tokens: list[OCRToken] = []
        for index, text in enumerate(text_values):
            if not str(text).strip():
                continue
            box = (
                self._box_from_points(box_values[index])
                if index < len(box_values)
                else [0.0, index * row_pitch, placeholder_width, (index + 1) * row_pitch]
            )
            confidence = score_values[index] if index < len(score_values) else 0.8
            if box is not None:
                tokens.append(OCRToken(str(text).strip(), box, self._score(confidence)))
        return sorted(tokens, key=lambda token: (token.box[1], token.box[0]))

    @staticmethod
    def _first_value(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _as_sequence(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return None

    @staticmethod
    def _box_from_points(points: Any) -> list[float] | None:
        try:
            values = [[float(point[0]), float(point[1])] for point in points]
            if len(values) < 2:
                return None
            xs = [point[0] for point in values]
            ys = [point[1] for point in values]
            return [min(xs), min(ys), max(xs), max(ys)]
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _replace_text_with_vietocr(self, tokens: list[OCRToken], gray_image: Any) -> list[OCRToken]:
        try:
            from PIL import Image
            import numpy as np

            result: list[OCRToken] = []
            height, width = gray_image.shape[:2]
            for token in tokens:
                x1, y1, x2, y2 = [int(round(value)) for value in token.box]
                x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
                y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
                crop = gray_image[y1:y2, x1:x2]
                if crop.size == 0:
                    result.append(token)
                    continue
                pil_crop = Image.fromarray(np.asarray(crop)).convert("RGB")
                recognized = str(self.vietocr_predictor.predict(pil_crop)).strip()
                result.append(OCRToken(recognized or token.text, token.box, token.confidence, source="vietocr"))
            return result
        except Exception:
            return tokens

    @staticmethod
    def _read_sidecar(path: Path, width: int = 1200, height: int = 2000) -> list[OCRToken]:
        sidecar = path.with_suffix(".txt")
        if not sidecar.exists():
            return []
        try:
            lines = [line.strip() for line in sidecar.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except (OSError, UnicodeError):
            return []
        # Demo-only synthetic boxes: distribute them across the full image
        # height so each box lands in the vertical band of its line, instead
        # of stacking in the top ~28px of the receipt.
        row_height = height / max(1, len(lines))
        return [
            OCRToken(
                text,
                [
                    0.0,
                    float(min(height - 1, index * row_height)),
                    float(width - 1),
                    float(min(height, (index + 1) * row_height)),
                ],
                0.55,
                source="sidecar",
            )
            for index, text in enumerate(lines)
        ]
