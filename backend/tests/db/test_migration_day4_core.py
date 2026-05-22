"""Day-4 core migration: file-content assertions (no live DB), mirroring the rag_chunks test."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_DIR = REPO_ROOT / "app" / "db" / "migrations" / "versions"


def _migration_text() -> str:
    files = list(VERSIONS_DIR.glob("*day4_core.py"))
    assert len(files) == 1, f"expected one day4_core migration, found {files}"
    return files[0].read_text(encoding="utf-8")


def test_creates_all_day4_tables() -> None:
    contents = _migration_text()
    for table in (
        "CREATE TABLE users",
        "CREATE TABLE invitations",
        "CREATE TABLE conversations",
        "CREATE TABLE messages",
        "CREATE TABLE widgets",
        "CREATE TABLE episodic_memories",
        "CREATE TABLE audit_log",
    ):
        assert table in contents, f"missing: {table}"


def test_episodic_memory_has_vector_and_audit_has_trace_columns() -> None:
    contents = _migration_text()
    assert "vector(384)" in contents  # episodic memory embedding
    assert "request_id" in contents and "trace_id" in contents  # audit traceability


def test_fk_and_unique_constraints_present() -> None:
    contents = _migration_text()
    assert "REFERENCES users(id)" in contents
    assert "REFERENCES conversations(id) ON DELETE CASCADE" in contents
    assert "CREATE UNIQUE INDEX ix_users_email" in contents
    assert "CREATE UNIQUE INDEX ix_invitations_token" in contents


def test_chains_onto_rag_chunks_and_downgrades() -> None:
    contents = _migration_text()
    assert 'revision: str = "0003"' in contents
    assert 'down_revision: str | None = "0002"' in contents
    # Downgrade drops users last (after dependent tables) to respect FK order.
    assert "DROP TABLE IF EXISTS users" in contents
