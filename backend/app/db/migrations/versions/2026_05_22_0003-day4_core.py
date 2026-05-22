"""day4 core tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-22

Creates the Day-4 runtime tables: ``users`` (matching the fastapi-users SQLAlchemyBaseUserTableUUID
shape + a ``role`` column + ``created_at``), ``invitations`` (admin invite flow), ``conversations``
and ``messages`` (chat persistence), ``widgets`` (embeddable widget config), ``episodic_memories``
(long-term episodic memory with a pgvector ``vector(384)`` embedding), and ``audit_log`` (one row
per sensitive action, e.g. every ``write_memory``). Raw SQL keeps DDL in one reviewable place and
matches the existing migration style; ``vector(384)`` is stable across the bge-small / MiniLM choice
(both 384-dim) and relies on the extension enabled in revision 0001.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE users (
            id              UUID PRIMARY KEY,
            email           VARCHAR(320) NOT NULL,
            hashed_password VARCHAR(1024) NOT NULL,
            is_active       BOOLEAN NOT NULL DEFAULT TRUE,
            is_superuser    BOOLEAN NOT NULL DEFAULT FALSE,
            is_verified     BOOLEAN NOT NULL DEFAULT FALSE,
            role            VARCHAR(16) NOT NULL DEFAULT 'user',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")

    op.execute(
        """
        CREATE TABLE invitations (
            id          UUID PRIMARY KEY,
            email       VARCHAR(320) NOT NULL,
            role        VARCHAR(16) NOT NULL DEFAULT 'user',
            token       VARCHAR(128) NOT NULL,
            created_by  UUID REFERENCES users(id) ON DELETE SET NULL,
            expires_at  TIMESTAMPTZ NOT NULL,
            used_at     TIMESTAMPTZ,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_invitations_token ON invitations (token)")

    op.execute(
        """
        CREATE TABLE conversations (
            id          UUID PRIMARY KEY,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_conversations_user ON conversations (user_id)")

    op.execute(
        """
        CREATE TABLE messages (
            id              UUID PRIMARY KEY,
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            VARCHAR(16) NOT NULL,
            content         TEXT NOT NULL,
            tool_name       VARCHAR(64),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_messages_conversation ON messages (conversation_id, created_at)")

    op.execute(
        """
        CREATE TABLE widgets (
            widget_id       VARCHAR(64) PRIMARY KEY,
            allowed_origins JSONB NOT NULL DEFAULT '[]'::jsonb,
            theme           VARCHAR(16) NOT NULL DEFAULT 'light',
            primary_color   VARCHAR(32) NOT NULL DEFAULT '#2563eb',
            position        VARCHAR(16) NOT NULL DEFAULT 'bottom-right',
            greeting        TEXT NOT NULL DEFAULT 'Hi! Ask me about this project''s issues.',
            enabled_tools   JSONB NOT NULL DEFAULT '[]'::jsonb,
            is_active       BOOLEAN NOT NULL DEFAULT TRUE,
            created_by      UUID REFERENCES users(id) ON DELETE SET NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE episodic_memories (
            id          UUID PRIMARY KEY,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            event_type  VARCHAR(32) NOT NULL,
            content     TEXT NOT NULL,
            subject     VARCHAR(200),
            embedding   vector(384),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_episodic_memories_user ON episodic_memories (user_id)")

    op.execute(
        """
        CREATE TABLE audit_log (
            id          UUID PRIMARY KEY,
            actor_id    UUID REFERENCES users(id) ON DELETE SET NULL,
            action      VARCHAR(64) NOT NULL,
            target_type VARCHAR(64),
            target_id   VARCHAR(128),
            request_id  VARCHAR(64),
            trace_id    VARCHAR(64),
            details     JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_audit_log_actor ON audit_log (actor_id)")
    op.execute("CREATE INDEX ix_audit_log_action ON audit_log (action)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP TABLE IF EXISTS episodic_memories")
    op.execute("DROP TABLE IF EXISTS widgets")
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("DROP TABLE IF EXISTS conversations")
    op.execute("DROP TABLE IF EXISTS invitations")
    op.execute("DROP TABLE IF EXISTS users")
