from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import get_settings
from .database import initialize_database
from .models import Invoice, InvoiceItem  # noqa: F401 - register ORM tables

settings = get_settings()
initialize_database()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Vietnamese invoice Intelligent Document Processing API",
)
origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api", tags=["IDP"])

if settings.processed_dir.exists():
    app.mount("/processed", StaticFiles(directory=settings.processed_dir), name="processed")
if settings.frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")


@app.get("/health", tags=["system"])
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "pipeline_mode": settings.pipeline_mode,
        "paddle_enabled": settings.use_paddle,
        "vietocr_enabled": settings.use_vietocr,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")
