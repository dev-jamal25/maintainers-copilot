from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single MetaData target for SQLAlchemy models and Alembic autogenerate."""
