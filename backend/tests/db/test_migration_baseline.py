from pathlib import Path

from app.db.base import Base

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
MIGRATIONS_DIR = REPO_ROOT / "app" / "db" / "migrations"
VERSIONS_DIR = MIGRATIONS_DIR / "versions"


def test_alembic_ini_script_location_points_at_migrations_package() -> None:
    contents = ALEMBIC_INI.read_text(encoding="utf-8")
    assert "script_location = app/db/migrations" in contents


def test_alembic_env_module_exists() -> None:
    assert (MIGRATIONS_DIR / "env.py").is_file()
    assert (MIGRATIONS_DIR / "script.py.mako").is_file()


def test_baseline_migration_enables_and_drops_vector_extension() -> None:
    baseline_files = list(VERSIONS_DIR.glob("*baseline_pgvector.py"))
    assert len(baseline_files) == 1, f"expected one baseline file, found {baseline_files}"
    contents = baseline_files[0].read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in contents
    assert "DROP EXTENSION IF EXISTS vector" in contents
    assert "down_revision: str | None = None" in contents


def test_base_metadata_has_no_tables_yet() -> None:
    assert Base.metadata.tables == {}
