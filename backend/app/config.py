from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the API and the optional IDP adapters."""

    app_name: str = "Vietnamese Invoice IDP"
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'idp_invoice.db').as_posix()}"
    pipeline_mode: Literal["heuristic", "vlm"] = "heuristic"
    vlm_provider: str = "mock"
    vlm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    use_paddle: bool = False
    use_vietocr: bool = False
    # Real-mode OCR: when use_paddle is True the sidecar/mock text fallback is
    # disabled and a paddle failure surfaces as an explicit error instead of
    # silently fabricating results. Set IDP_OCR_STRICT=false only for the
    # offline demo flow.
    ocr_strict: bool = True
    paddle_lang: str = "vi"
    # PaddleOCR 3.x model family version. "PP-OCRv6" is the newest official
    # family with a Vietnamese-capable latin dictionary and supports
    # tiny|small|medium sizes; "PP-OCRv5" supports mobile|server. Pin it for
    # reproducible demo runs or when a version has a regression.
    paddle_ocr_version: str = "PP-OCRv6"
    # Model size within the family. Bigger is slower but reads Vietnamese
    # diacritics better. See IDP_PADDLE_OCR_VERSION for valid values.
    paddle_model_size: str = "medium"
    # "auto" prefers CUDA and falls back to CPU when the GPU is unavailable;
    # "gpu" forces GPU (fails fast without the GPU runtime); "cpu" skips GPU.
    paddle_device: str = "auto"
    # Recognition batch size — raise while the GPU has free VRAM, lower it when
    # training/fine-tuning shares the same card.
    paddle_rec_batch_size: int = 16
    # Longest side forwarded to the detector; larger images are internally
    # scaled down, bounding VRAM on the 8 GB card.
    paddle_det_limit_side_len: int = 960
    # Paddle's fraction of free GPU memory (0.5 keeps headroom for the desktop
    # compositor on a laptop GPU). Values outside (0, 1) fall back to 0.5.
    paddle_gpu_memory_fraction: float = 0.5
    model_dir: Path = PROJECT_ROOT / "models"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    max_upload_bytes: int = 12_000_000
    max_image_dimension: int = 3_000
    max_decoded_pixels: int = 40_000_000
    review_confidence_threshold: float = 0.85
    cors_origins: str = "*"
    allowed_extensions: str = ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff"

    model_config = SettingsConfigDict(
        env_file=(".env", str(PROJECT_ROOT / ".env")),
        env_prefix="IDP_",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            extension.strip().lower() if extension.strip().startswith(".") else f".{extension.strip().lower()}"
            for extension in self.allowed_extensions.split(",")
            if extension.strip()
        }

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_directories(self) -> None:
        for directory in (self.model_dir, self.upload_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _resolve_directory(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_sqlite_url(url: str) -> str:
    """Make relative SQLite paths independent of the process working directory."""
    if not url.startswith("sqlite:///"):
        return url
    raw_path = url[len("sqlite:///") :]
    if raw_path.startswith("/") or (len(raw_path) >= 2 and raw_path[1] == ":"):
        return url
    path = (PROJECT_ROOT / raw_path).resolve()
    return f"sqlite:///{path.as_posix()}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.model_dir = _resolve_directory(settings.model_dir)
    settings.upload_dir = _resolve_directory(settings.upload_dir)
    settings.processed_dir = _resolve_directory(settings.processed_dir)
    settings.database_url = _resolve_sqlite_url(settings.database_url)
    settings.ensure_directories()
    return settings
