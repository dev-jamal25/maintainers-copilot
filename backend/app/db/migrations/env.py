"""Alembic environment.

Reads the URL from app settings (not from alembic.ini), uses the async
engine via run_sync, and binds against `Base.metadata` so future
autogenerate revisions detect model changes correctly.
"""

import asyncio

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import Settings
from app.db.base import Base

config = context.config

target_metadata = Base.metadata

_settings = Settings()
config.set_main_option("sqlalchemy.url", _settings.database.database_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
