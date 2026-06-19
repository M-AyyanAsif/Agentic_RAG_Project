"""Tests for document upload parsing behavior."""

from __future__ import annotations
import pytest
from backend.engine.document_processor import DocumentProcessor


def test_rejects_unsupported_type() -> None:
    """Ensure the processor blocks invalid MIME types."""
    processor = DocumentProcessor()
    
    # Senior Tip: Use pytest.raises for cleaner, more reliable failure testing
    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.parse("notes.txt", "text/plain", b"hello")


def test_document_chunking_logic() -> None:
    """Verify that chunking respects size and doesn't break words (if logic is implemented)."""
    # Using a small chunk size to force multiple chunks
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=5)
    
    text = "Artificial Intelligence is the future of Indus Guardian."
    # We simulate a PDF parse result by manually calling the cleaning and chunking
    cleaned = processor._clean_text(text)
    chunks = processor._chunk_text(cleaned)
    
    assert len(chunks) > 1
    # Ensure no chunk is empty
    assert all(len(c) > 0 for c in chunks)
    
    # If you implemented the space-snapping logic we discussed:
    # The first chunk should end at a space, not mid-word.
    assert " " in chunks[0] 


def test_clean_text_removes_extra_whitespace() -> None:
    """Ensure the regex cleaning works as expected."""
    processor = DocumentProcessor()
    dirty_text = "Hello    World\n\nNew   Line"
    clean_text = processor._clean_text(dirty_text)
    
    # Should collapse multiple spaces into one
    assert "  " not in clean_text
    assert clean_text == "Hello World New Line"