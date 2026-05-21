from __future__ import annotations

from app.repositories import chunk_repository as repo

# Note: live cosine retrieval is exercised against a real Postgres+pgvector instance
# (docker compose) — not in unit CI, which has no DB service. These tests cover the pure
# vector encoding and the SQL shape that the queries depend on.


def test_encode_vector_uses_pgvector_text_format() -> None:
    assert repo.encode_vector([1, 2, 3]) == "[1.00000000,2.00000000,3.00000000]"
    encoded = repo.encode_vector([0.5, -1.25])
    assert encoded.startswith("[") and encoded.endswith("]")


def test_search_sql_uses_cosine_distance_operator() -> None:
    sql = str(repo._SEARCH_SQL)
    assert "rag_chunks" in sql
    assert "<=>" in sql
    assert "level = :level" in sql


def test_insert_sql_targets_rag_chunks_with_vector_cast() -> None:
    sql = str(repo._INSERT_SQL)
    assert "INSERT INTO rag_chunks" in sql
    assert "CAST(:embedding AS vector)" in sql
