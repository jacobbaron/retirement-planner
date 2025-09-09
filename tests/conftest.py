"""
Pytest configuration and shared fixtures for the retirement planner tests.
"""

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.models import User


@pytest.fixture(scope="function")
def db_engine():
    """Create a PostgreSQL database for testing."""
    # Use the same database URL as the main app but with a test database
    db_url = os.getenv(
        "DB_URL",
        "postgresql://retirement_user:retirement_pass@localhost:5432/retirement_planner",
    )
    # Replace the database name with a test database
    db_url = db_url.replace("/retirement_planner", "/retirement_planner_test")

    engine = create_engine(db_url)

    # Drop and recreate all tables for clean test state
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    return engine


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(email="test@example.com", name="Test User")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user
