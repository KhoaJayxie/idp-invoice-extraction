from __future__ import annotations

import os

# SQLAlchemy's C extension is optional; disabling it keeps the Windows demo
# usable on machines where loading optional compiled DLLs is restricted.
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def initialize_database() -> None:
    """Create tables and apply tiny additive migrations for local demos.

    A full Alembic setup is appropriate for production deployments. The demo
    deliberately keeps startup dependency-light, so newly added nullable fields
    are added safely when an existing SQLite file is reused.
    """
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name != "sqlite":
        return
    expected_columns = {
        "invoices": {
            "aligned_image_url": "VARCHAR(512)",
            "processed_image_url": "VARCHAR(512)",
            "binary_image_url": "VARCHAR(512)",
        },
    }
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name not in existing:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    )


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
