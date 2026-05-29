from sqlalchemy import create_engine, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.config.settings import settings

# SQLite compatibility monkeypatch for PostgreSQL-specific JSONB type
import sqlalchemy.dialects.postgresql as pg
pg.JSONB = JSON

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy database models"""
    pass

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # checks connection health before issuing queries
)

# Create sessionmaker for local database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency generator for database sessions.
    Ensures that the session is always closed after request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
