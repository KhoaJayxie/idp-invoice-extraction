from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from functools import lru_cache
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session, selectinload

from ..config import get_settings
from ..database import get_db
from ..idp_engine.hybrid import HybridPipeline
from ..idp_engine.pipeline import HeuristicPipeline, IDPPipeline, VLMPipeline
from ..idp_engine.ocr_detector import OCRUnavailableError
from ..idp_engine.postprocessor import validate_invoice
from ..idp_engine.preprocessor import ImagePreprocessor, PreprocessConfig, ImagePreprocessingError
from ..models import Invoice, InvoiceItem
from ..schemas import (
    DuplicateMatch,
    DuplicateResponse,
    ExtractionResponse,
    InvoiceCreate,
    InvoiceOut,
    InvoiceUpdate,
    LineItem,
    PreprocessOptions,
    PreviewResponse,
    ReviewAction,
    StatsResponse,
    ValidationIssue,
    ValidationRequest,
)

router = APIRouter()
settings = get_settings()


def _safe_filename(filename: str | None) -> tuple[str, str]:
    original = Path(filename or "invoice.jpg").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original).stem).strip("._") or "invoice"
    suffix = Path(original).suffix.lower()
    if suffix and suffix not in settings.allowed_extension_set:
        raise HTTPException(status_code=415, detail="Định dạng ảnh không được hỗ trợ.")
    if not suffix:
        suffix = ".jpg"
    return original[:255], f"{stem[:80]}{suffix}"


def _parse_options(raw: str | None) -> PreprocessOptions:
    if not raw:
        return PreprocessOptions()
    try:
        return PreprocessOptions.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"preprocess_options không hợp lệ: {exc}") from exc


async def _read_limited(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = settings.max_upload_bytes + 1 - total
        chunk = await file.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File vượt giới hạn {settings.max_upload_bytes / 1_000_000:.1f} MB.",
            )
    return b"".join(chunks)


def _validate_upload_size(raw: bytes) -> None:
    if not raw:
        raise HTTPException(status_code=422, detail="File upload rỗng.")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File vượt giới hạn {settings.max_upload_bytes / 1_000_000:.1f} MB.",
        )



def _save_upload(raw: bytes, filename: str | None, ocr_text: str | None = None) -> tuple[Path, str]:
    original_filename, safe_name = _safe_filename(filename)
    target = settings.upload_dir / f"{uuid4().hex}_{safe_name}"
    target.write_bytes(raw)
    if ocr_text and ocr_text.strip():
        target.with_suffix(".txt").write_text(ocr_text.strip() + "\n", encoding="utf-8")
    return target, original_filename


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validation_dicts(issues: list[ValidationIssue]) -> list[dict]:
    return [issue.model_dump(mode="json") for issue in issues]


def _payload_presence(payload: InvoiceCreate | InvoiceUpdate) -> dict[str, bool]:
    fields = ("timestamp", "seller", "invoice_id", "subtotal", "tax", "discount", "total_cost")
    explicit = dict(getattr(payload, "field_presence", None) or {})
    return {
        field: bool(explicit.get(field, field in payload.model_fields_set and getattr(payload, field, None) is not None))
        for field in fields
    }


def _item_presence(item: LineItem) -> dict[str, bool]:
    explicit = dict(item.field_presence or {})
    return {
        field: bool(explicit.get(field, field in item.model_fields_set and getattr(item, field, None) is not None))
        for field in ("quantity", "unit_price", "amount")
    }


def _validation_invoice(row: Invoice) -> InvoiceData:
    """Build the same validation input for extracted and manually edited data."""
    from ..schemas.invoice import InvoiceData

    return InvoiceData(
        invoice_id=row.invoice_id,
        invoice_series=row.invoice_series,
        invoice_form=row.invoice_form,
        seller=row.seller,
        buyer=row.buyer,
        address=row.address,
        buyer_address=row.buyer_address,
        tax_id=row.tax_id,
        buyer_tax_id=row.buyer_tax_id,
        currency=row.currency,
        timestamp=row.timestamp,
        subtotal=row.subtotal,
        tax=row.tax,
        discount=row.discount,
        total_cost=row.total_cost,
        discount_rate=row.discount_rate,
        tax_rate=row.tax_rate,
        tax_included=row.tax_included,
        rounding=row.rounding,
        field_presence=row.field_presence or {},
        items=[
            LineItem(
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                confidence=item.confidence,
                bbox=item.bbox,
                field_presence=item.field_presence or {},
            )
            for item in row.items
        ],
    )


def _find_duplicates(db: Session, *, file_hash: str | None, invoice_id: str | None, seller: str | None,
                     timestamp: datetime | None, total_cost: float | None, exclude_id: int | None = None) -> list[dict]:
    from ..idp_engine.postprocessor import normalize_text

    matches: list[dict] = []
    query = db.query(Invoice)
    if exclude_id is not None:
        query = query.filter(Invoice.id != exclude_id)
    if file_hash:
        for row in query.filter(Invoice.file_hash == file_hash).all():
            matches.append({"row": row, "reason": "Trùng SHA-256 file ảnh"})
    if invoice_id and seller:
        norm_seller = normalize_text(seller).casefold()
        for row in query.filter(Invoice.invoice_id == invoice_id).all():
            if row.seller and normalize_text(row.seller).casefold() == norm_seller and row.id not in {item["row"].id for item in matches}:
                matches.append({"row": row, "reason": "Trùng mã hóa đơn và nhà bán"})
    if seller and timestamp and total_cost is not None:
        norm_seller = normalize_text(seller).casefold()
        for row in query.filter(Invoice.seller.isnot(None)).all():
            same_day = row.timestamp and row.timestamp.date() == timestamp.date()
            same_total = abs(float(row.total_cost or 0) - float(total_cost)) <= max(0.01, abs(float(total_cost)) * 0.001)
            if row.seller and normalize_text(row.seller).casefold() == norm_seller and same_day and same_total and row.id not in {item["row"].id for item in matches}:
                matches.append({"row": row, "reason": "Trùng nhà bán, ngày và tổng tiền"})
    return matches


def _duplicate_out(match: dict) -> DuplicateMatch:
    row = match["row"]
    return DuplicateMatch(
        id=row.id,
        reason=match["reason"],
        invoice_id=row.invoice_id,
        seller=row.seller,
        total_cost=float(row.total_cost or 0),
        status=row.status,
        created_at=row.created_at,
    )
def _revalidate_row(row: Invoice, *, override: bool | None = None, override_reason: str | None = None, reset_status: bool = True):
    result = validate_invoice(_validation_invoice(row), float(row.confidence or 0.0), settings.review_confidence_threshold)
    row.validation_issues = _validation_dicts(result.issues)
    if override is not None:
        if override and not (override_reason or "").strip():
            raise HTTPException(status_code=422, detail="Cần ghi rõ lý do override validation.")
        row.validation_override = bool(override)
        row.validation_override_reason = (override_reason or "").strip() or None
    if reset_status and result.critical_errors and row.status in {"VERIFIED", "VALIDATED"} and not row.validation_override:
        row.status = "NEEDS_REVIEW"
        row.reviewed_at = None
    return result
def _prediction_snapshot(payload: InvoiceCreate) -> dict:
    """Return an immutable, JSON-safe copy of the AI draft."""
    if payload.original_prediction:
        return payload.original_prediction
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "data": payload.model_dump(mode="json", exclude={"original_prediction"}),
    }


def _invoice_snapshot(row: Invoice) -> dict:
    """Serialize editable invoice values for correction-history diffs."""
    values = {
        key: getattr(row, key)
        for key in (
            "invoice_id", "invoice_series", "invoice_form", "seller", "buyer",
            "address", "buyer_address", "tax_id", "buyer_tax_id", "currency",
            "timestamp", "subtotal", "tax", "discount", "discount_rate", "total_cost", "tax_rate",
            "tax_included", "rounding", "field_presence",
        )
    }
    values["timestamp"] = values["timestamp"].isoformat() if values["timestamp"] else None
    values["items"] = [
        {
            "name": item.name,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "amount": item.amount,
            "field_presence": item.field_presence or {},
        }
        for item in row.items
    ]
    return values


def _append_history(row: Invoice, event: dict) -> None:
    history = list(row.correction_history or [])
    history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
    row.correction_history = history


def _to_out(row: Invoice) -> InvoiceOut:
    return InvoiceOut(
        id=row.id,
        invoice_id=row.invoice_id,
        seller=row.seller,
        buyer=row.buyer,
        address=row.address,
        buyer_address=row.buyer_address,
        tax_id=row.tax_id,
        buyer_tax_id=row.buyer_tax_id,
        currency=row.currency,
        timestamp=row.timestamp,
        subtotal=row.subtotal,
        tax=row.tax,
        discount=row.discount,
        total_cost=row.total_cost,
        discount_rate=row.discount_rate,
        tax_rate=row.tax_rate,
        tax_included=row.tax_included,
        rounding=row.rounding,
        field_presence=row.field_presence or {},
        items=[
            LineItem(
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                confidence=item.confidence,
                bbox=item.bbox,
                field_presence=item.field_presence or {},
            )
            for item in row.items
        ],
        field_confidence=(row.raw_json or {}).get("field_confidence", {}),
        status=row.status,
        confidence=row.confidence,
        strategy=row.strategy,
        source_image=row.source_image,
        aligned_image_url=row.aligned_image_url,
        processed_image_url=row.processed_image_url,
        binary_image_url=row.binary_image_url,
        original_filename=row.original_filename,
        file_hash=row.file_hash,
        source_image_url=f"/api/uploads/{row.source_image}" if row.source_image else None,
        raw_json=row.raw_json,
        preprocessing_json=row.preprocessing_json,
        original_prediction=row.original_prediction or {},
        validation_issues=[ValidationIssue.model_validate(issue) for issue in (row.validation_issues or [])],
        validation_override=bool(row.validation_override),
        validation_override_reason=row.validation_override_reason,
        duplicate_override=bool(row.duplicate_override),
        duplicate_override_reason=row.duplicate_override_reason,
        correction_history=row.correction_history or [],
        reviewed_at=row.reviewed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@lru_cache(maxsize=4)
def _get_pipeline(mode: str) -> IDPPipeline:
    if mode == "vlm":
        return VLMPipeline(settings.vlm_provider, settings.openai_api_key or settings.gemini_api_key, settings.vlm_model)
    if mode == "hybrid":
        return HybridPipeline()
    return HeuristicPipeline()


def _pipeline_for(mode: str | None) -> IDPPipeline:
    selected = (mode or settings.pipeline_mode).lower()
    if selected == "ocr_kie":  # explicit alias of the default OCR+KIE mode
        selected = "heuristic"
    if selected not in {"heuristic", "vlm", "hybrid"}:
        raise HTTPException(status_code=400, detail="mode phải là heuristic (ocr_kie), vlm hoặc hybrid.")
    return _get_pipeline(selected)


@router.post("/preprocess/preview", response_model=PreviewResponse)
async def preprocess_preview(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
):
    raw = await _read_limited(file)
    _validate_upload_size(raw)
    preprocess_options = _parse_options(options)
    processor = ImagePreprocessor(
        PreprocessConfig(
            max_dimension=preprocess_options.max_dimension or settings.max_image_dimension,
            max_decoded_pixels=settings.max_decoded_pixels,
        ),
        max_bytes=settings.max_upload_bytes,
    )
    try:
        result = processor.process(raw, preprocess_options)
    except (ImagePreprocessingError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PreviewResponse(
        original=result.encode_preview("original"),
        enhanced=result.encode_preview("enhanced"),
        binary=result.encode_preview("binary"),
        metadata=result.metadata,
    )


@router.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(...),
    mode: str | None = Form(default=None),
    ocr_text: str | None = Form(default=None),
    preprocess_options: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    raw = await _read_limited(file)
    _validate_upload_size(raw)
    file_hash = _file_sha256(raw)
    preprocess_options = _parse_options(preprocess_options)
    pipeline = _pipeline_for(mode)
    if ocr_text and len(ocr_text) > 100_000:
        raise HTTPException(status_code=413, detail="Văn bản OCR demo vượt giới hạn 100 KB.")
    target, original_filename = _save_upload(raw, file.filename, ocr_text)
    try:
        response = pipeline.run(
            target,
            options=preprocess_options,
            source_name=target.name,
        )
    except OCRUnavailableError as exc:
        target.unlink(missing_ok=True)
        target.with_suffix(".txt").unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"OCR engine chưa sẵn sàng (real mode không dùng sidecar/mock): {exc}",
        ) from exc
    except (ImagePreprocessingError, ValueError, RuntimeError, OSError) as exc:
        target.unlink(missing_ok=True)
        target.with_suffix(".txt").unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}") from exc
    except Exception as exc:  # unexpected engine/image errors: 500 + cleanup
        target.unlink(missing_ok=True)
        target.with_suffix(".txt").unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý không mong muốn: {exc}") from exc
    source_url = f"/api/uploads/{target.name}"
    matches = _find_duplicates(
        db,
        file_hash=file_hash,
        invoice_id=response.invoice.invoice_id,
        seller=response.invoice.seller,
        timestamp=response.invoice.timestamp,
        total_cost=response.invoice.total_cost,
    )
    return response.model_copy(
        update={
            "source_image": target.name,
            "file_hash": file_hash,
            "original_filename": original_filename,
            "source_image_url": source_url,
            "duplicates": [_duplicate_out(match) for match in matches],
        }
    )


@router.get("/uploads/{filename}")
def uploaded_image(filename: str):
    candidate = (settings.upload_dir / Path(filename).name).resolve()
    root = settings.upload_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File không tồn tại.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File không tồn tại.")
    return FileResponse(candidate)


@router.get("/ocr/status")
def ocr_status() -> dict:
    """Report the configured OCR engine without forcing model warm-up.

    When PaddleOCR is enabled the first call may take tens of seconds while
    models initialize; the response distinguishes that state clearly so the
    UI can explain why extraction is slower on the first run. The VLM block
    reports hybrid-fallback readiness (cheap checks only — never loads the
    model).
    """
    detector = _get_pipeline("heuristic").detector  # type: ignore[attr-defined]
    vlm_available, vlm_reason = _hybrid_vlm_readiness()
    return {
        "use_paddle": settings.use_paddle,
        "use_vietocr": settings.use_vietocr,
        "strict": settings.ocr_strict,
        "engine": detector.status(),
        "warning": detector.backend_warning,
        "vlm_provider": settings.vlm_provider,
        "vlm_local_model": settings.vlm_local_model,
        "vlm_available": vlm_available,
        "vlm_unavailable_reason": vlm_reason,
    }


def _hybrid_vlm_readiness() -> tuple[bool, str | None]:
    """Cheap hybrid-fallback probe: config + deps + model cache, no warm-up."""
    if (settings.vlm_provider or "mock").lower() == "mock":
        return False, "IDP_VLM_PROVIDER=mock — đặt 'local' để bật VLM cục bộ"
    from ..idp_engine.vlm_extractor import VLMExtractor

    extractor = VLMExtractor(
        settings.vlm_provider,
        settings.openai_api_key or settings.gemini_api_key,
        settings.vlm_model,
    )
    ok, reason = extractor.fallback_available()
    return ok, reason


@router.get("/invoices", response_model=list[InvoiceOut])
def list_invoices(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    search: str | None = Query(default=None, max_length=100),
    db: Session = Depends(get_db),
):
    query = db.query(Invoice).options(selectinload(Invoice.items)).order_by(Invoice.created_at.desc())
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        query = query.filter((Invoice.seller.ilike(pattern)) | (Invoice.invoice_id.ilike(pattern)))
    return [_to_out(row) for row in query.offset(skip).limit(limit).all()]


@router.get("/invoices/{invoice_pk}", response_model=InvoiceOut)
def get_invoice(invoice_pk: int, db: Session = Depends(get_db)):
    row = db.query(Invoice).options(selectinload(Invoice.items)).filter(Invoice.id == invoice_pk).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _to_out(row)


@router.post("/invoices", response_model=InvoiceOut, status_code=201)
def create_invoice(payload: InvoiceCreate, db: Session = Depends(get_db)):
    duplicate_matches = _find_duplicates(
        db,
        file_hash=payload.file_hash,
        invoice_id=payload.invoice_id,
        seller=payload.seller,
        timestamp=payload.timestamp,
        total_cost=payload.total_cost,
    )
    if duplicate_matches and not payload.duplicate_override:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DUPLICATE_INVOICE",
                "message": "Hóa đơn có khả năng trùng với dữ liệu đã lưu.",
                "duplicates": [_duplicate_out(match).model_dump(mode="json") for match in duplicate_matches],
            },
        )
    if duplicate_matches and not (payload.duplicate_override_reason or "").strip():
        raise HTTPException(status_code=422, detail="Cần ghi rõ lý do override khi bỏ qua hóa đơn trùng.")
    data = payload.model_dump(exclude={"items", "field_confidence", "original_prediction", "validation_issues"})
    data["original_prediction"] = _prediction_snapshot(payload)
    data["correction_history"] = []
    data["validation_issues"] = [issue.model_dump(mode="json") for issue in payload.validation_issues]
    data["field_presence"] = _payload_presence(payload)
    data["raw_json"] = {**(data.get("raw_json") or {}), "field_confidence": payload.field_confidence}
    if data.get("preprocessing_json") is None:
        data.pop("preprocessing_json", None)
    row = Invoice(**data)
    row.items = [
        InvoiceItem(
            line_no=index,
            name=item.name,
            quantity=item.quantity,
            unit_price=item.unit_price,
            amount=item.amount,
            confidence=item.confidence,
            bbox=item.bbox,
            field_presence=_item_presence(item),
        )
        for index, item in enumerate(payload.items, 1)
    ]
    result = _revalidate_row(
        row,
        override=payload.validation_override,
        override_reason=payload.validation_override_reason,
        reset_status=False,
    )
    row.duplicate_override = bool(payload.duplicate_override)
    row.duplicate_override_reason = (payload.duplicate_override_reason or "").strip() or None
    if result.critical_errors and payload.status in {"VERIFIED", "VALIDATED"} and not payload.validation_override:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_BLOCKED",
            "message": "Không thể xác minh khi còn lỗi validation nghiêm trọng.",
            "issues": _validation_dicts(result.critical_errors),
        })
    if duplicate_matches:
        _append_history(row, {"type": "duplicate_override", "reason": row.duplicate_override_reason, "matches": [_duplicate_out(match).model_dump(mode="json") for match in duplicate_matches]})
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(row)
    return _to_out(row)


@router.put("/invoices/{invoice_pk}", response_model=InvoiceOut)
def update_invoice(invoice_pk: int, payload: InvoiceUpdate, db: Session = Depends(get_db)):
    row = db.query(Invoice).options(selectinload(Invoice.items)).filter(Invoice.id == invoice_pk).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    before = _invoice_snapshot(row)
    previous_status = row.status
    values = payload.model_dump(
        exclude_unset=True,
        exclude={"items", "field_confidence", "original_prediction", "validation_issues", "validation_override", "validation_override_reason", "duplicate_override", "duplicate_override_reason"},
    )
    if "raw_json" in values and values["raw_json"] is None:
        values.pop("raw_json")
    for key, value in values.items():
        setattr(row, key, value)
    presence = dict(row.field_presence or {})
    for field in ("timestamp", "seller", "invoice_id", "subtotal", "tax", "discount", "total_cost"):
        if field in payload.model_fields_set:
            presence[field] = getattr(payload, field) is not None
    if payload.field_presence:
        presence.update(payload.field_presence)
    row.field_presence = presence
    if payload.original_prediction is not None and not row.original_prediction:
        row.original_prediction = payload.original_prediction
    if payload.validation_override is not None:
        row.validation_override = payload.validation_override
        row.validation_override_reason = (payload.validation_override_reason or "").strip() or None
    if payload.duplicate_override is not None:
        row.duplicate_override = payload.duplicate_override
        row.duplicate_override_reason = (payload.duplicate_override_reason or "").strip() or None
    if payload.field_confidence is not None:
        raw = dict(row.raw_json or {})
        raw["field_confidence"] = payload.field_confidence
        row.raw_json = raw
    if payload.items is not None:
        row.items.clear()
        row.items.extend(
            InvoiceItem(
                line_no=index,
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                confidence=item.confidence,
                bbox=item.bbox,
                field_presence=item.field_presence,
            )
            for index, item in enumerate(payload.items, 1)
        )
    after = _invoice_snapshot(row)
    if before != after:
        _append_history(row, {"type": "correction", "before": before, "after": after})
    result = _revalidate_row(
        row,
        override=payload.validation_override,
        override_reason=payload.validation_override_reason,
        reset_status=True,
    )
    if payload.status in {"VERIFIED", "VALIDATED"} and result.critical_errors and not row.validation_override:
        db.rollback()
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_BLOCKED",
            "message": "Không thể xác minh khi còn lỗi validation nghiêm trọng.",
            "issues": _validation_dicts(result.critical_errors),
        })
    if payload.status is not None and payload.status != previous_status:
        _append_history(row, {"type": "status_change", "from_status": previous_status, "status": payload.status})
        if payload.status in {"VERIFIED", "REJECTED"}:
            row.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.post("/invoices/{invoice_pk}/review", response_model=InvoiceOut)
def review_invoice(invoice_pk: int, payload: ReviewAction, db: Session = Depends(get_db)):
    """Apply an explicit human review decision and persist an audit event."""
    row = db.query(Invoice).options(selectinload(Invoice.items)).filter(Invoice.id == invoice_pk).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    previous = row.status
    result = _revalidate_row(
        row,
        override=payload.validation_override,
        override_reason=payload.validation_override_reason,
        reset_status=False,
    )
    if payload.status in {"VERIFIED"} and result.critical_errors and not row.validation_override:
        db.rollback()
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_BLOCKED",
            "message": "Không thể xác minh khi còn lỗi validation nghiêm trọng.",
            "issues": _validation_dicts(result.critical_errors),
        })
    row.status = payload.status
    if payload.validation_override:
        _append_history(row, {"type": "validation_override", "reason": row.validation_override_reason})
    row.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    _append_history(row, {
        "type": "review_action",
        "from_status": previous,
        "status": payload.status,
        "note": payload.note,
    })
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.post("/invoices/{invoice_pk}/validate", response_model=InvoiceOut)
def revalidate_invoice(invoice_pk: int, payload: ValidationRequest, db: Session = Depends(get_db)):
    row = db.query(Invoice).options(selectinload(Invoice.items)).filter(Invoice.id == invoice_pk).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    result = _revalidate_row(row, override=payload.validation_override, override_reason=payload.validation_override_reason)
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.get("/invoices/{invoice_pk}/duplicates", response_model=DuplicateResponse)
def invoice_duplicates(invoice_pk: int, db: Session = Depends(get_db)):
    row = db.get(Invoice, invoice_pk)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    matches = _find_duplicates(
        db,
        file_hash=row.file_hash,
        invoice_id=row.invoice_id,
        seller=row.seller,
        timestamp=row.timestamp,
        total_cost=row.total_cost,
        exclude_id=row.id,
    )
    return DuplicateResponse(duplicates=[_duplicate_out(match) for match in matches])
@router.get("/invoices/{invoice_pk}/review-history")
def invoice_review_history(invoice_pk: int, db: Session = Depends(get_db)):
    row = db.get(Invoice, invoice_pk)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return row.correction_history or []


@router.delete("/invoices/{invoice_pk}", status_code=204)
def delete_invoice(invoice_pk: int, db: Session = Depends(get_db)):
    row = db.get(Invoice, invoice_pk)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    db.delete(row)
    db.commit()


@router.get("/stats", response_model=StatsResponse)
def stats(
    days: str = Query(default="all"),
    db: Session = Depends(get_db),
):
    rows = db.query(Invoice).all()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if days == "today":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = [row for row in rows if (row.timestamp or row.created_at) >= cutoff]
    elif days == "month":
        cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        rows = [row for row in rows if (row.timestamp or row.created_at) >= cutoff]
    elif days.isdigit():
        cutoff = now - timedelta(days=max(1, min(int(days), 3650)))
        rows = [row for row in rows if (row.timestamp or row.created_at) >= cutoff]
    elif days != "all":
        raise HTTPException(status_code=422, detail="days phải là all, today, month hoặc số ngày.")
    by_seller: dict[str, dict[str, float | int | str]] = defaultdict(lambda: {"seller": "", "total": 0.0, "count": 0})
    by_month: dict[str, dict[str, float | int | str]] = defaultdict(lambda: {"month": "", "total": 0.0, "count": 0})
    for row in rows:
        seller = row.seller or "Không rõ"
        seller_data = by_seller[seller]
        seller_data["seller"] = seller
        seller_data["total"] = float(seller_data["total"]) + float(row.total_cost or 0)
        seller_data["count"] = int(seller_data["count"]) + 1
        month = row.timestamp.strftime("%Y-%m") if row.timestamp else row.created_at.strftime("%Y-%m")
        month_data = by_month[month]
        month_data["month"] = month
        month_data["total"] = float(month_data["total"]) + float(row.total_cost or 0)
        month_data["count"] = int(month_data["count"]) + 1
    return StatsResponse(
        invoice_count=len(rows),
        total_spend=sum(float(row.total_cost or 0) for row in rows),
        total_tax=sum(float(row.tax or 0) for row in rows),
        needs_review=sum(row.status == "NEEDS_REVIEW" for row in rows),
        validated=sum(row.status in {"VALIDATED", "VERIFIED"} for row in rows),
        by_seller=sorted(by_seller.values(), key=lambda value: float(value["total"]), reverse=True),
        by_month=sorted(by_month.values(), key=lambda value: str(value["month"])),
    )


def _spreadsheet_safe(value):
    if isinstance(value, str) and value.lstrip(" \t\r\n")[:1] in {"=", "+", "-", "@"}:
        return f"'{value}"
    return value


@router.get("/export")
def export_invoices(
    format: Literal["csv", "xlsx"] = Query(default="csv"),
    db: Session = Depends(get_db),
):
    rows = db.query(Invoice).options(selectinload(Invoice.items)).order_by(Invoice.created_at.desc()).all()
    headers = ["id", "invoice_id", "seller", "address", "timestamp", "subtotal", "tax", "tax_rate", "discount", "discount_rate", "rounding", "total_cost", "currency", "file_hash", "status", "validation_errors", "confidence", "items"]
    values = [
        [
            row.id,
            _spreadsheet_safe(row.invoice_id or ""),
            _spreadsheet_safe(row.seller or ""),
            _spreadsheet_safe(row.address or ""),
            row.timestamp.isoformat(sep=" ") if row.timestamp else "",
            row.subtotal,
            row.tax,
            row.tax_rate,
            row.discount,
            row.discount_rate,
            row.rounding,
            row.total_cost,
            _spreadsheet_safe(row.currency or ""),
            row.file_hash or "",
            row.status,
            _spreadsheet_safe("; ".join(issue.get("message", "") for issue in (row.validation_issues or []))),
            row.confidence,
            _spreadsheet_safe("; ".join(f"{item.name} x{item.quantity} = {item.amount}" for item in row.items)),
        ]
        for row in rows
    ]
    if format == "csv":
        stream = io.StringIO()
        stream.write("﻿")
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(values)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=invoices.csv"},
        )
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Cài openpyxl để xuất XLSX.") from exc
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Invoices"
    sheet.append(headers)
    for value in values:
        sheet.append(value)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=invoices.xlsx"},
    )
