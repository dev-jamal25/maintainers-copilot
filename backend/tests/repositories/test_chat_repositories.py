"""SQL-shape tests for the chat repositories (live DB behaviour is exercised under compose)."""

from __future__ import annotations

from app.repositories import conversation_repository as conv
from app.repositories import message_repository as msg


def test_conversation_get_is_user_scoped() -> None:
    sql = str(conv._GET_SQL)
    assert "FROM conversations" in sql
    assert "id = :id AND user_id = :user_id" in sql  # no cross-user reads


def test_conversation_insert_targets_table() -> None:
    assert "INSERT INTO conversations" in str(conv._INSERT_SQL)


def test_message_list_is_ordered_by_time() -> None:
    sql = str(msg._LIST_SQL)
    assert "FROM messages" in sql
    assert "WHERE conversation_id = :conversation_id" in sql
    assert "ORDER BY created_at" in sql


def test_message_insert_targets_table() -> None:
    assert "INSERT INTO messages" in str(msg._INSERT_SQL)
