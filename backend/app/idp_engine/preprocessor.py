from __future__ import annotations

"""Production-oriented image preparation for Vietnamese receipt photographs.

The preprocessor deliberately returns every useful intermediate instead of only
one grayscale image. This makes it possible to inspect failures in the UI and
keeps OCR coordinates tied to the exact image sent to the detector.
"""

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - exercised only in minimal imports
    cv2 = None
    np = None
    Image = None
    ImageOps = None

from ..schemas.invoice import PreprocessMetadata


class ImagePreprocessingError(ValueError):
    """Raised when an upload is not a decodable, supported image."""


@dataclass(slots=True)
class PreprocessConfig:
    auto_perspective: bool = True
    deskew: bool = True
    illumination_correction: bool = True
    denoise: bool = True
    clahe: bool = True
    sharpen: bool = True
    binarize: bool = True
    max_dimension: int = 3000
    min_document_area_ratio: float = 0.12
    max_deskew_angle: float = 18.0
    jpeg_quality: int = 88
    max_decoded_pixels: int = 40_000_000


@dataclass(slots=True)
class PreprocessResult:
    """All image variants and metadata produced for one upload."""

    original_bgr: Any
    enhanced_gray: Any
    binary: Any
    metadata: PreprocessMetadata
    source_path: Path | None = None
    transform_matrix: Any = None

    @property
    def width(self) -> int:
        return int(self.enhanced_gray.shape[1])

    @property
    def height(self) -> int:
        return int(self.enhanced_gray.shape[0])

    @property
    def ocr_image(self) -> Any:
        return self.enhanced_gray

    def encode_preview(self, variant: str = "enhanced", quality: int = 88) -> str:
        image = {
            "original": self.original_bgr,
            "enhanced": self.enhanced_gray,
            "binary": self.binary,
        }.get(variant)
        if image is None:
            raise ValueError(f"Unknown image variant: {variant}")
        return encode_data_url(image, quality=quality, png=variant == "binary")


class ImagePreprocessor:
    """Normalize, rectify and enhance receipt images with safe fallbacks.

    A failed page contour is not a fatal error: the original oriented image is
    retained and the warning is surfaced to the reviewer. This is important for
    thermal receipts whose border is often missing or blends into the table.
    """

    def __init__(self, config: PreprocessConfig | None = None, max_bytes: int = 12_000_000):
        self.config = config or PreprocessConfig()
        self.max_bytes = max_bytes

    def process(
        self,
        source: str | Path | bytes | bytearray | Any,
        options: PreprocessConfig | dict[str, Any] | Any | None = None,
    ) -> PreprocessResult:
        self._require_dependencies()
        config = self._merge_options(options)
        source_path = Path(source) if isinstance(source, (str, Path)) else None
        image, original_width, original_height = self._load_image(source, config.max_decoded_pixels)
        image, scale = self._resize_to_limit(image, config.max_dimension)
        warnings: list[str] = []
        corners: list[list[float]] | None = None
        transform_matrix = None
        document_detected = False
        document_confidence = 0.0
        perspective_applied = False

        if config.auto_perspective:
            quad, document_confidence = self.detect_document(image, config.min_document_area_ratio)
            if quad is not None:
                warped, transform_matrix = self._warp_document(image, quad)
                if warped is not None:
                    image = warped
                    corners = [[float(x), float(y)] for x, y in quad]
                    document_detected = True
                    perspective_applied = True
                else:
                    warnings.append("Không thể warp phối cảnh; giữ nguyên ảnh đã xoay EXIF.")
            else:
                warnings.append("Không phát hiện được biên tờ hóa đơn rõ ràng; dùng toàn bộ khung ảnh.")
        else:
            warnings.append("Đã tắt tự động cắt/căn chỉnh phối cảnh.")

        # A very large perspective warp can exceed the configured safety bound.
        image, post_warp_scale = self._resize_to_limit(image, config.max_dimension)
        scale *= post_warp_scale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        deskew_angle = 0.0
        deskew_applied = False
        if config.deskew:
            angle = self._estimate_skew(gray)
            if angle is not None and abs(angle) >= 0.15:
                if abs(angle) <= config.max_deskew_angle:
                    image = self._rotate_bound(image, angle)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    deskew_angle = float(angle)
                    deskew_applied = True
                else:
                    warnings.append(f"Góc nghiêng {angle:.1f}° vượt giới hạn an toàn; không xoay.")
        else:
            warnings.append("Đã tắt deskew tự động.")

        if config.illumination_correction:
            gray = self._correct_illumination(gray)
        else:
            warnings.append("Đã tắt khử bóng/chuẩn hóa ánh sáng.")
        illumination_applied = config.illumination_correction

        if config.denoise:
            gray = self._denoise(gray)
        denoise_applied = config.denoise

        if config.clahe:
            gray = self._clahe(gray)
        clahe_applied = config.clahe

        if config.sharpen:
            gray = self._unsharp_mask(gray)
        sharpen_applied = config.sharpen

        enhanced = np.clip(gray, 0, 255).astype(np.uint8)
        if config.binarize:
            binary = self._binarize(enhanced)
        else:
            binary = np.zeros_like(enhanced)
            warnings.append("Đã tắt tạo ảnh nhị phân.")
        binarization_applied = config.binarize

        blur_score = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
        brightness_mean = float(np.mean(enhanced))
        contrast_std = float(np.std(enhanced))
        if blur_score < 25:
            warnings.append("Ảnh có thể bị mờ (variance Laplacian thấp).")
        if contrast_std < 18:
            warnings.append("Độ tương phản thấp; nên chụp lại hoặc bật tăng cường tương phản.")
        if brightness_mean < 35 or brightness_mean > 235:
            warnings.append("Độ sáng ảnh nằm ngoài vùng khuyến nghị.")
        if document_confidence and document_confidence < 0.35:
            warnings.append("Độ tin cậy phát hiện biên hóa đơn thấp.")

        metadata = PreprocessMetadata(
            original_width=original_width,
            original_height=original_height,
            processed_width=int(enhanced.shape[1]),
            processed_height=int(enhanced.shape[0]),
            scale=float(scale),
            document_detected=document_detected,
            document_confidence=float(document_confidence),
            perspective_applied=perspective_applied,
            corners=corners,
            deskew_applied=deskew_applied,
            deskew_angle=deskew_angle,
            denoise_applied=denoise_applied,
            illumination_correction_applied=illumination_applied,
            clahe_applied=clahe_applied,
            sharpen_applied=sharpen_applied,
            binarization_applied=binarization_applied,
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
            warnings=warnings,
        )
        # Keep a color copy after geometric transforms and before enhancement;
        # overlay coordinates now match this image exactly.
        display_original = image.copy()
        return PreprocessResult(
            original_bgr=display_original,
            enhanced_gray=enhanced,
            binary=binary,
            metadata=metadata,
            source_path=source_path,
            transform_matrix=transform_matrix,
        )

    def detect_document(
        self, image: Any, min_area_ratio: float | None = None
    ) -> tuple[Any | None, float]:
        """Find the best convex quadrilateral around a page/receipt.

        Detection runs on a small copy for speed, then maps corners back to the
        input image. The score combines area and rectangularity, avoiding a tiny
        text contour being mistaken for the document.
        """
        self._require_dependencies()
        min_area_ratio = min_area_ratio or self.config.min_document_area_ratio
        height, width = image.shape[:2]
        detect_limit = 1600
        detect_scale = min(1.0, detect_limit / max(height, width))
        if detect_scale < 1.0:
            small = cv2.resize(image, (int(width * detect_scale), int(height * detect_scale)), interpolation=cv2.INTER_AREA)
        else:
            small = image
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        low = int(max(10, 0.66 * median))
        high = int(min(255, max(low + 30, 1.33 * median)))
        edges = cv2.Canny(blurred, low, high)
        kernel_size = max(3, int(round(min(small.shape[:2]) / 180)) * 2 + 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = float(small.shape[0] * small.shape[1])
        best: tuple[float, Any, float] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            if area_ratio < min_area_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                # A stricter epsilon can recover corners on rounded receipt
                # edges without accepting arbitrary text contours.
                approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            points = approx.reshape(4, 2).astype(np.float32)
            ordered = self._order_points(points)
            width_a = np.linalg.norm(ordered[2] - ordered[3])
            width_b = np.linalg.norm(ordered[1] - ordered[0])
            height_a = np.linalg.norm(ordered[1] - ordered[2])
            height_b = np.linalg.norm(ordered[0] - ordered[3])
            rect_area = max(1.0, float(max(width_a, width_b) * max(height_a, height_b)))
            rectangularity = min(1.0, area / rect_area)
            score = min(1.0, area_ratio) * (0.65 + 0.35 * rectangularity)
            if best is None or score > best[0]:
                best = (score, ordered, area_ratio)
        if best is None:
            return None, 0.0
        _, points, area_ratio = best
        if detect_scale < 1.0:
            points = points / detect_scale
        return points.astype(np.float32), float(min(1.0, area_ratio * 1.2))

    def _load_image(
        self,
        source: str | Path | bytes | bytearray | Any,
        max_decoded_pixels: int | None = None,
    ) -> tuple[Any, int, int]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists() or not path.is_file():
                raise ImagePreprocessingError(f"Không tìm thấy ảnh: {path}")
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise ImagePreprocessingError(f"Không đọc được ảnh: {path}") from exc
            if len(raw) > self.max_bytes:
                raise ImagePreprocessingError(
                    f"Ảnh vượt giới hạn {self.max_bytes / 1_000_000:.1f} MB."
                )
            return self._decode_bytes(raw, max_decoded_pixels)
        if isinstance(source, (bytes, bytearray)):
            if len(source) > self.max_bytes:
                raise ImagePreprocessingError(
                    f"Ảnh vượt giới hạn {self.max_bytes / 1_000_000:.1f} MB."
                )
            return self._decode_bytes(bytes(source), max_decoded_pixels)
        if np is not None and isinstance(source, np.ndarray):
            image = source.copy()
            if image.size == 0:
                raise ImagePreprocessingError("Ảnh rỗng.")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.shape[2] != 3:
                raise ImagePreprocessingError("Ảnh ndarray phải có 1, 3 hoặc 4 kênh.")
            height, width = image.shape[:2]
            pixel_limit = max_decoded_pixels or self.config.max_decoded_pixels
            if height * width > pixel_limit:
                raise ImagePreprocessingError(
                    f"Ảnh có quá nhiều điểm ảnh ({width}×{height}); giới hạn là {pixel_limit:,}."
                )
            return self._ensure_uint8(image), width, height
        raise ImagePreprocessingError("Nguồn ảnh phải là path, bytes hoặc numpy.ndarray.")

    def _decode_bytes(
        self,
        raw: bytes,
        max_decoded_pixels: int | None = None,
    ) -> tuple[Any, int, int]:
        try:
            with Image.open(BytesIO(raw)) as pil_image:
                width, height = pil_image.size
                pixel_limit = max_decoded_pixels or self.config.max_decoded_pixels
                if width * height > pixel_limit:
                    raise ImagePreprocessingError(
                        f"Ảnh có quá nhiều điểm ảnh ({width}×{height}); giới hạn là {pixel_limit:,}."
                    )
                oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
                rgb = np.asarray(oriented)
        except ImagePreprocessingError:
            raise
        except Exception as exc:
            raise ImagePreprocessingError("File không phải ảnh hợp lệ hoặc ảnh bị hỏng.") from exc
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ImagePreprocessingError("Không thể chuyển ảnh về RGB.")
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        if width < 2 or height < 2:
            raise ImagePreprocessingError("Kích thước ảnh quá nhỏ.")
        return self._ensure_uint8(image), width, height

    def _resize_to_limit(self, image: Any, max_dimension: int) -> tuple[Any, float]:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= max_dimension:
            return image, 1.0
        scale = max_dimension / float(longest)
        resized = cv2.resize(image, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _warp_document(self, image: Any, points: Any) -> tuple[Any | None, Any | None]:
        ordered = self._order_points(points)
        tl, tr, br, bl = ordered
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        if width < 32 or height < 32:
            return None, None
        # Avoid a portrait receipt becoming an accidentally microscopic output.
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), target)
        warped = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return warped, matrix

    @staticmethod
    def _order_points(points: Any) -> Any:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).ravel()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    def _estimate_skew(self, gray: Any) -> float | None:
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        border = max(2, int(min(gray.shape[:2]) * 0.02))
        threshold[:border, :] = 0
        threshold[-border:, :] = 0
        threshold[:, :border] = 0
        threshold[:, -border:] = 0
        foreground = float(np.count_nonzero(threshold)) / threshold.size
        if foreground < 0.002 or foreground > 0.55:
            return None
        coordinates = np.column_stack(np.where(threshold > 0))
        if coordinates.shape[0] < 30:
            return None
        rect = cv2.minAreaRect(coordinates.astype(np.float32))
        raw_angle = float(rect[-1])
        angle = -(90.0 + raw_angle) if raw_angle < -45.0 else -raw_angle
        # A perfectly axis-aligned page yields ±90°; that is "not skewed", not
        # an extreme tilt, so ignore it instead of warning about a clamp.
        if abs(angle) >= 85.0:
            return None
        return float(angle)

    @staticmethod
    def _rotate_bound(image: Any, angle: float) -> Any:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        new_width = int((height * sine) + (width * cosine))
        new_height = int((height * cosine) + (width * sine))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(255, 255, 255))

    @staticmethod
    def _correct_illumination(gray: Any) -> Any:
        height, width = gray.shape[:2]
        min_side = min(height, width)
        # The background estimate is low-frequency by construction, so the
        # expensive morphological close runs on a downscaled copy and is
        # upscaled before the division — visually identical and ~10x faster
        # on full-resolution scans.
        scale = min(1.0, 720.0 / max(height, width))
        if scale < 1.0:
            small = cv2.resize(gray, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = gray
        kernel_size = max(15, min(101, int(min_side / 8) | 1))
        kernel_size = max(3, round(kernel_size * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (0, 0), max(1.0, min(small.shape[:2]) / 40.0))
        if scale < 1.0:
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
        corrected = cv2.divide(gray, np.maximum(background, 1), scale=255)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def _denoise(gray: Any) -> Any:
        if min(gray.shape[:2]) < 32:
            return cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def _clahe(gray: Any) -> Any:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _unsharp_mask(gray: Any) -> Any:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1)
        return cv2.addWeighted(gray, 1.45, blurred, -0.45, 0)

    @staticmethod
    def _binarize(gray: Any) -> Any:
        min_side = min(gray.shape[:2])
        block = max(11, min(71, int(min_side / 18) | 1))
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            11,
        )
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    def _merge_options(self, options: PreprocessConfig | dict[str, Any] | Any | None) -> PreprocessConfig:
        if options is None:
            return PreprocessConfig(**asdict(self.config))
        if isinstance(options, PreprocessConfig):
            return options
        values = asdict(self.config)
        if isinstance(options, dict):
            values.update({key: value for key, value in options.items() if value is not None})
        else:
            for key in values:
                if hasattr(options, key):
                    value = getattr(options, key)
                    if value is not None:
                        values[key] = value
        return PreprocessConfig(**values)

    @staticmethod
    def _ensure_uint8(image: Any) -> Any:
        if image.dtype == np.uint8:
            return image
        if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _require_dependencies() -> None:
        if cv2 is None or np is None or Image is None:
            raise RuntimeError("Image preprocessing requires opencv-python-headless, numpy and Pillow.")


def encode_data_url(image: Any, quality: int = 88, png: bool = False) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to encode preview images.")
    if image is None:
        raise ValueError("Cannot encode an empty image.")
    if png:
        success, encoded = cv2.imencode(".png", image)
        mime = "image/png"
    else:
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        mime = "image/jpeg"
    if not success:
        raise ImagePreprocessingError("Không thể mã hóa ảnh preview.")
    import base64

    return f"data:{mime};base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"
