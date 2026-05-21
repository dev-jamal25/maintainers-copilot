from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_DIR = REPO_ROOT / "app" / "db" / "migrations" / "versions"


def test_rag_chunks_migration_creates_table_with_vector_column() -> None:
    files = list(VERSIONS_DIR.glob("*rag_chunks.py"))
    assert len(files) == 1, f"expected one rag_chunks migration, found {files}"
    contents = files[0].read_text(encoding="utf-8")
    assert "CREATE TABLE rag_chunks" in contents
    assert "vector(384)" in contents
    assert "DROP TABLE IF EXISTS rag_chunks" in contents


def test_rag_chunks_migration_chains_onto_baseline() -> None:
    files = list(VERSIONS_DIR.glob("*rag_chunks.py"))
    contents = files[0].read_text(encoding="utf-8")
    assert 'revision: str = "0002"' in contents
    assert 'down_revision: str | None = "0001"' in contents
