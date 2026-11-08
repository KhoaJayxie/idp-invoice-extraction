from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
from ..idp_engine.pipeline import HeuristicPipeline, IDPPipeline, VLMPipeline
from ..idp_engine.ocr_detector import OCRUnavailableError
from ..idp_engine.preprocessor import ImagePreprocessor, PreprocessConfig, ImagePreprocessingError
from ..models import Invoice, InvoiceItem
from ..schemas import (
    ExtractionResponse,
    InvoiceCreate,
    InvoiceOut,
    InvoiceUpdate,
    LineItem,
    PreprocessOptions,
    PreviewResponse,
    StatsResponse,
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
        items=[
            LineItem(
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                confidence=item.confidence,
                bbox=item.bbox,
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
        source_image_url=f"/api/uploads/{row.source_image}" if row.source_image else None,
        raw_json=row.raw_json,
        preprocessing_json=row.preprocessing_json,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@lru_cache(maxsize=4)
def _get_pipeline(mode: str) -> IDPPipeline:
    if mode == "vlm":
        return VLMPipeline(settings.vlm_provider, settings.openai_api_key or settings.gemini_api_key, settings.vlm_model)
    return HeuristicPipeline()


def _pipeline_for(mode: str | None) -> IDPPipeline:
    selected = (mode or settings.pipeline_mode).lower()
    if selected not in {"heuristic", "vlm"}:
        raise HTTPException(status_code=400, detail="mode phải là heuristic hoặc vlm.")
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
):
    raw = await _read_limited(file)
    _validate_upload_size(raw)
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
    return response.model_copy(
        update={
            "source_image": target.name,
            "original_filename": original_filename,
            "source_image_url": source_url,
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
    UI can explain why extraction is slower on the first run.
    """
    detector = _get_pipeline("heuristic").detector  # type: ignore[attr-defined]
    return {
        "use_paddle": settings.use_paddle,
        "use_vietocr": settings.use_vietocr,
        "strict": settings.ocr_strict,
        "engine": detector.status(),
        "warning": detector.backend_warning,
    }


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
    data = payload.model_dump(exclude={"items", "field_confidence"})
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
        )
        for index, item in enumerate(payload.items, 1)
    ]
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.put("/invoices/{invoice_pk}", response_model=InvoiceOut)
def update_invoice(invoice_pk: int, payload: InvoiceUpdate, db: Session = Depends(get_db)):
    row = db.query(Invoice).options(selectinload(Invoice.items)).filter(Invoice.id == invoice_pk).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    values = payload.model_dump(exclude_unset=True, exclude={"items", "field_confidence"})
    if "raw_json" in values and values["raw_json"] is None:
        values.pop("raw_json")
    for key, value in values.items():
        setattr(row, key, value)
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
            )
            for index, item in enumerate(payload.items, 1)
        )
    db.commit()
    db.refresh(row)
    return _to_out(row)


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
        validated=sum(row.status == "VALIDATED" for row in rows),
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
    headers = ["id", "invoice_id", "seller", "address", "timestamp", "subtotal", "tax", "discount", "total_cost", "status", "confidence", "items"]
    values = [
        [
            row.id,
            _spreadsheet_safe(row.invoice_id or ""),
            _spreadsheet_safe(row.seller or ""),
            _spreadsheet_safe(row.address or ""),
            row.timestamp.isoformat(sep=" ") if row.timestamp else "",
            row.subtotal,
            row.tax,
            row.discount,
            row.total_cost,
            row.status,
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
