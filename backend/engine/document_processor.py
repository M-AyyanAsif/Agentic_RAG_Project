"""
Document parsing and Pinecone Indexing (RAG-ready version).
- Production Optimized: Tailored for gemini-embedding-2 with safe string formatting.
"""

from __future__ import annotations
import io
import logging
import re
import time
from dataclasses import dataclass
from typing import List
import docx
from pypdf import PdfReader
import google.generativeai as genai
from pinecone import Pinecone

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

import time
from google.api_core.exceptions import ResourceExhausted

def generate_embeddings(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        # A 4-second exponential backoff fallback mechanism
        retries = 3
        delay = 4 
        
        while retries > 0:
            try:
                # Your actual embedding invocation line:
                response = genai.embed_content(
                    model="models/gemini-embedding-2",
                    content=chunk
                )
                embeddings.append(response['embedding'])
                
                # --- THE FIX: FORCED RPM COOLDOWN ---
                # 47 chunks * 2 seconds = ~94 seconds total processing time.
                # This keeps your workflow completely safe from RPM bans.
                time.sleep(2.0) 
                break
                
            except ResourceExhausted as e:
                logger.warning(f"Rate limit hit on chunk {i}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                retries -= 1
                
        if retries == 0:
            raise Exception(f"Failed processing chunk {i} due to quota exhaustion.")
            
    return embeddings

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
        self.settings = get_settings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Google GenAI
        genai.configure(api_key=self.settings.google_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = self.pc.Index(self.settings.pinecone_index_name)

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

    def index_to_pinecone(self, session_id: str, parsed_doc: ParsedDocument):
        try:
            vectors_to_upsert = []
            logger.info(f"Generating embeddings for {len(parsed_doc.chunks)} chunks using gemini-embedding-2...")

            for i, chunk in enumerate(parsed_doc.chunks):
                retries = 3
                delay = 1.0
                res = None
                
                # CRITICAL V2 FIX: Use prefix instructions inside the text. No title -> title: none
                formatted_content = f"title: none | text: {chunk}"
                
                for attempt in range(retries):
                    try:
                        # task_type is REMOVED. output_dimensionality keeps vector at 1024
                        res = genai.embed_content(
                            model="gemini-embedding-2",
                            content=formatted_content,
                            output_dimensionality=1024
                        )
                        break  
                    except Exception as exc:
                        err_msg = str(exc)
                        if "429" in err_msg or "Resource exhausted" in err_msg:
                            if attempt == retries - 1:
                                raise exc
                            logger.warning(f"Rate limit hit on chunk {i}. Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= 2  
                        else:
                            raise exc  

                if res and 'embedding' in res:
                    embedding = res['embedding']
                    vectors_to_upsert.append({
                        "id": f"{session_id}_{i}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "filename": parsed_doc.filename,
                            "session_id": session_id
                        }
                    })
                
                # Pacing gap to prevent hitting cloud thresholds
                time.sleep(0.25)

            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert, namespace=session_id)
                logger.info(f"Success: Indexed {len(parsed_doc.chunks)} chunks in namespace: {session_id}")
            
        except Exception as e:
            logger.error(f"Pinecone indexing failed: {e}")
            raise e

    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("PDF parsing failed: %s", str(e))
            return ""

    def _parse_docx(self, file_bytes: bytes) -> str:
        try:
            document = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error("DOCX parsing failed: %s", str(e))
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text) 
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space != -1: end = last_space
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if len(c) > 10]