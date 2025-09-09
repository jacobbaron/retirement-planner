"""
Database base configuration and session management.

This module provides the SQLAlchemy base class, engine, and session management
for the retirement planner application.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

from app.config import get_settings

# Create the declarative base
Base = declarative_base()

# Global variables for engine and session factory
_engine = None
_session_factory = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.db_url,
            echo=settings.app_env == "development",  # Log SQL in development
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


def get_session() -> Session:
    """Get a new database session."""
    session_factory = get_session_factory()
    return session_factory()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI/Flask to get database sessions."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def create_tables():
    """Create all tables in the database."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables in the database."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
