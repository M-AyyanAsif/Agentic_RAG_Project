"""Tests for document upload parsing behavior."""

from __future__ import annotations

from backend.engine.document_processor import DocumentProcessor


def test_rejects_unsupported_type() -> None:
    processor = DocumentProcessor()
    try:
        processor.parse("notes.txt", "text/plain", b"hello")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported type")
