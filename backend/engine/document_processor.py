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
    """
    Parses uploaded documents into clean, chunked text
    ready for RAG (Retrieval-Augmented Generation).
    """

    SUPPORTED_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    def __init__(self, chunk_size: int = 800):
        self.chunk_size = chunk_size

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

    # ---------------- PDF ----------------
    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            logger.info("Parsed PDF with %d pages", len(pages))
            return "\n".join(pages)

        except Exception as e:
            logger.error("PDF parsing failed: %s", str(e))
            return ""

    # ---------------- DOCX ----------------
    def _parse_docx(self, file_bytes: bytes) -> str:
        try:
            document = docx.Document(io.BytesIO(file_bytes))

            paragraphs = [
                p.text for p in document.paragraphs
                if p.text and p.text.strip()
            ]

            logger.info("Parsed DOCX with %d paragraphs", len(paragraphs))
            return "\n".join(paragraphs)

        except Exception as e:
            logger.error("DOCX parsing failed: %s", str(e))
            return ""

    # ---------------- CLEANING ----------------
    def _clean_text(self, text: str) -> str:
        """
        Normalize document text:
        - remove extra spaces
        - fix newlines
        - remove noise
        """
        text = re.sub(r"\s+", " ", text)  # collapse spaces
        text = text.replace("\n", " ")   # normalize newlines
        return text.strip()

    # ---------------- CHUNKING (VERY IMPORTANT FOR RAG) ----------------
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embeddings / retrieval.
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end

        return chunks