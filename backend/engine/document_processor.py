"""Document parsing for PDF and DOCX inputs (RAG-ready version)."""

from __future__ import annotations
import io
import logging
import re
from dataclasses import dataclass
from typing import List
import docx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class ParsedDocument:
    filename: str
    content: str
    content_type: str
    chunks: List[str]

class DocumentProcessor:
    SUPPORTED_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap # Senior Tip: Always use overlap

    def parse(self, filename: str, content_type: str, file_bytes: bytes) -> ParsedDocument:
        if content_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {content_type}")

        if content_type == "application/pdf":
            text = self._parse_pdf(file_bytes)
        else:
            text = self._parse_docx(file_bytes)

        cleaned_text = self._clean_text(text)
        chunks = self._chunk_text(cleaned_text)

        return ParsedDocument(
            filename=filename,
            content=cleaned_text,
            content_type=content_type,
            chunks=chunks,
        )

    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            logger.info("Parsed PDF with %d pages", len(pages))
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("PDF parsing failed: %s", str(e))
            return ""

    def _parse_docx(self, file_bytes: bytes) -> str:
        try:
            document = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            logger.info("Parsed DOCX with %d paragraphs", len(paragraphs))
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error("DOCX parsing failed: %s", str(e))
            return ""

    def _clean_text(self, text: str) -> str:
        # Senior Tip: Clean extra spaces but keep single newlines for structure
        text = re.sub(r"[ \t]+", " ", text) 
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """Recursive-style chunking with overlap to preserve meaning."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            
            # If not at the end of the doc, try to snap to the nearest space
            # so we don't cut a word in half.
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space

            chunks.append(text[start:end].strip())
            
            # Move start forward, but subtract overlap
            start = end - self.chunk_overlap
            
            # Safety check to avoid infinite loops
            if start >= text_len or end >= text_len:
                break
                
        return [c for c in chunks if len(c) > 10] # Filter out tiny noise chunks