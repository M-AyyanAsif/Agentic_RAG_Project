"""SQLite helpers for chat history and semantic cache."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ChatRecord:
    session_id: str
    role: str
    content: str
    created_at: str


class DatabaseManager:
    """Manages SQLite storage for chat history and semantic cache."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_session
                ON chat_history(session_id);

                CREATE INDEX IF NOT EXISTS idx_docs_session
                ON session_documents(session_id);
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.sha256(
            query.strip().lower().encode("utf-8")
        ).hexdigest()

    # ---------------- CHAT HISTORY ----------------

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_history(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, self._now_iso()),
            )

    def get_messages(self, session_id: str) -> list[ChatRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, role, content, created_at
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [ChatRecord(**dict(row)) for row in rows]

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a chat session (hard delete)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM session_documents WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def list_sessions(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM chat_history
                GROUP BY session_id
                ORDER BY MAX(id) DESC
                """
            ).fetchall()

        return [row["session_id"] for row in rows]

    # ---------------- SEMANTIC CACHE ----------------

    def get_cached_answer(self, query: str, ttl_seconds: int) -> str | None:
        query_hash = self._hash_query(query)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT answer, created_at
                FROM semantic_cache
                WHERE query_hash = ?
                """,
                (query_hash,),
            ).fetchone()

        if not row:
            return None

        created_at = datetime.fromisoformat(row["created_at"])
        if created_at < cutoff:
            return None

        return str(row["answer"])

    def upsert_cached_answer(self, query: str, answer: str) -> None:
        query_hash = self._hash_query(query)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache(query_hash, question, answer, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    answer = excluded.answer,
                    created_at = excluded.created_at
                """,
                (query_hash, query, answer, self._now_iso()),
            )

    # ---------------- DOCUMENT STORAGE ----------------

    def add_document(self, session_id: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_documents(session_id, content, created_at)
                VALUES (?, ?, ?)
                """,
                (session_id, content, self._now_iso()),
            )

    def get_documents(self, session_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content
                FROM session_documents
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [str(row["content"]) for row in rows]