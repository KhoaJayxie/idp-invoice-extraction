from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import _get_pipeline
from backend.app.config import get_settings
from backend.app.database import Base, get_db
from backend.app.main import app


@pytest.fixture(scope="session", autouse=True)
def isolated_test_database(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Keep API tests from writing rows/files into the demo runtime directories.

    Tests also pin the offline demo OCR settings so the suite stays
    deterministic on machines where PaddleOCR is installed and `.env` enables
    real mode (strict real mode intentionally ignores sidecar text).
    """
    runtime = get_settings()
    original_upload_dir = runtime.upload_dir
    original_processed_dir = runtime.processed_dir
    original_use_paddle = runtime.use_paddle
    original_use_vietocr = runtime.use_vietocr
    runtime.upload_dir = tmp_path_factory.mktemp("uploads")
    runtime.processed_dir = tmp_path_factory.mktemp("processed")
    runtime.use_paddle = False
    runtime.use_vietocr = False
    _get_pipeline.cache_clear()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def override_get_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
    engine.dispose()
    runtime.upload_dir = original_upload_dir
    runtime.processed_dir = original_processed_dir
    runtime.use_paddle = original_use_paddle
    runtime.use_vietocr = original_use_vietocr
    _get_pipeline.cache_clear()
