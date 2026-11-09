from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Box(BaseModel):
    """An OCR/KIE rectangle in processed-image pixel coordinates."""

    text: str = ""
    box: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str | None = None
    source: str = "ocr"


class LineItem(BaseModel):
    name: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    amount: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)


class InvoiceData(BaseModel):
    invoice_id: str | None = None
    invoice_series: str | None = None
    invoice_form: str | None = None
    seller: str | None = None
    buyer: str | None = None
    address: str | None = None
    buyer_address: str | None = None
    tax_id: str | None = None
    buyer_tax_id: str | None = None
    currency: str | None = None
    timestamp: datetime | None = None
    subtotal: float = 0.0
    tax: float = 0.0
    discount: float = 0.0
    total_cost: float = 0.0
    items: list[LineItem] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)


class InvoiceCreate(InvoiceData):
    status: str = "NEEDS_REVIEW"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: str | None = None
    source_image: str | None = None
    aligned_image_url: str | None = None
    processed_image_url: str | None = None
    binary_image_url: str | None = None
    original_filename: str | None = None
    raw_json: dict[str, Any] | None = None
    preprocessing_json: dict[str, Any] | None = None


class InvoiceUpdate(BaseModel):
    invoice_id: str | None = None
    invoice_series: str | None = None
    invoice_form: str | None = None
    seller: str | None = None
    buyer: str | None = None
    address: str | None = None
    buyer_address: str | None = None
    tax_id: str | None = None
    buyer_tax_id: str | None = None
    currency: str | None = None
    timestamp: datetime | None = None
    subtotal: float | None = None
    tax: float | None = None
    discount: float | None = None
    total_cost: float | None = None
    items: list[LineItem] | None = None
    status: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    strategy: str | None = None
    source_image: str | None = None
    aligned_image_url: str | None = None
    processed_image_url: str | None = None
    binary_image_url: str | None = None
    original_filename: str | None = None
    raw_json: dict[str, Any] | None = None
    preprocessing_json: dict[str, Any] | None = None
    field_confidence: dict[str, float] | None = None


class InvoiceOut(InvoiceCreate):
    id: int
    source_image_url: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)


class MathCheck(BaseModel):
    line_sum: float = 0.0
    tax: float = 0.0
    discount: float = 0.0
    expected_total: float = 0.0
    actual_total: float = 0.0
    delta: float = 0.0
    passed: bool = False
    tolerance: float = 1.0


class ValidationResultSchema(BaseModel):
    status: str
    confidence: float
    warnings: list[str] = Field(default_factory=list)
    math: MathCheck


class PreprocessOptions(BaseModel):
    auto_perspective: bool = True
    deskew: bool = True
    illumination_correction: bool = True
    denoise: bool = True
    clahe: bool = True
    sharpen: bool = True
    binarize: bool = True
    max_dimension: int | None = Field(default=None, ge=256, le=8000)


class PreprocessMetadata(BaseModel):
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale: float = 1.0
    document_detected: bool = False
    document_confidence: float = 0.0
    perspective_applied: bool = False
    corners: list[list[float]] | None = None
    deskew_applied: bool = False
    deskew_angle: float = 0.0
    denoise_applied: bool = False
    illumination_correction_applied: bool = False
    clahe_applied: bool = False
    sharpen_applied: bool = False
    binarization_applied: bool = False
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    invoice: InvoiceData
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    boxes: list[Box] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    strategy: str
    raw_text: str = ""
    math: MathCheck | None = None
    preprocess: PreprocessMetadata | None = None
    source_image: str | None = None
    original_filename: str | None = None
    source_image_url: str | None = None
    aligned_image_url: str | None = None
    processed_image_url: str | None = None
    binary_image_url: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    engine: dict[str, Any] | None = None
    # Per-field KIE breakdown: {field: {value, kie_conf, ocr_conf, method,
    # box_ids}} — kie_conf combines the underlying OCR confidence with how
    # strong the structural evidence was (label match, spatial fallback...),
    # deliberately separate from the raw OCR confidence above.
    field_details: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PreviewResponse(BaseModel):
    original: str
    enhanced: str
    binary: str
    metadata: PreprocessMetadata


class StatsResponse(BaseModel):
    invoice_count: int
    total_spend: float
    total_tax: float
    needs_review: int
    validated: int
    by_seller: list[dict[str, Any]] = Field(default_factory=list)
    by_month: list[dict[str, Any]] = Field(default_factory=list)
