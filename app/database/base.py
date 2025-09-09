"""
Database base configuration and session management.

This module provides the SQLAlchemy base class, engine, and session management
for the retirement planner application.
"""

from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

# Create the declarative base
Base = declarative_base()

# Global variables for engine and session factory
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.db_url,
            echo=settings.app_env == "development",  # Log SQL in development
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


def get_session() -> Session:
    """Get a new database session."""
    session_factory = get_session_factory()
    return session_factory()  # type: ignore[no-any-return]


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI/Flask to get database sessions."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    """Create all tables in the database."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all tables in the database."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
