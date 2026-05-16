"""Hybrid retrieval service (Optimized for Google Cloud Embeddings)."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional
import google.generativeai as genai
from pinecone import Pinecone
import json
from typing import Any
from backend.core.config import Settings

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class RetrievalResult:
    source: str
    content: str
    score: float

class RetrievalService:
    """
    RAG Retrieval using Google Generative AI for embeddings.
    No local models = Zero RAM usage for embeddings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # Initialize Google GenAI
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize Pinecone
        self._pinecone = (
            Pinecone(api_key=settings.pinecone_api_key)
            if settings.pinecone_api_key else None
        )
        self._index = (
            self._pinecone.Index(settings.pinecone_index_name)
            if self._pinecone else None
        )

    def _get_embedding(self, text: str) -> List[float]:
        """Fetch vector from Google API."""
        try:
            # Senior Tip: Ensure we use the latest embedding model
            result = genai.embed_content(
                model="models/gemini-embedding-2", 
                content=text,
                task_type="retrieval_query",
                output_dimensionality=1024
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Google Embedding failed: {e}")
            return []

    def retrieve_hybrid(self, query: str, session_id: Optional[str] = None, top_k: int = 5) -> list[RetrievalResult]:
        """
        Queries Pinecone using the Google-generated vector.
        Uses session_id as the namespace to ensure document isolation.
        """
        if not self._index:
            logger.warning("Pinecone Index not initialized. Skipping retrieval.")
            return []

        # 1. Get the vector for the user's question
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []

        # 2. Query Pinecone with Namespace support
        try:
            res = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=session_id # CRITICAL: This must match your document_processor namespace
            )
        except Exception as e:
            logger.error(f"Pinecone Query failed: {e}")
            return []

        matches = res.get("matches", [])
        
        # 3. Filter and Format
        # We lowered the score threshold slightly to 0.3 to be more inclusive during testing
        results = []
        for m in matches:
            score = float(m.get("score", 0.0))
            if score > 0.3:
                # Handle cases where 'text' might be stored under 'content' in metadata
                metadata = m.get("metadata", {})
                content = metadata.get("text") or metadata.get("content") or ""
                
                results.append(RetrievalResult(
                    source="pinecone",
                    content=content,
                    score=score,
                ))

        logger.info(f"Retrieved {len(results)} relevant chunks from Pinecone (Namespace: {session_id}).")
        return results

    def format_context(self, results: list[Any]) -> str:
        """
        Converts retrieval results into a clean, structurally separated 
        string window for the LLM prompt, stripping away raw list-dict patterns.
        """
        if not results:
            return ""
        
        formatted_chunks = []
        for r in results:
            # Safely extract content whether it's an object, a dict, or wrapped inside a text block list
            content = ""
            if hasattr(r, "content"):
                content = r.content
            elif isinstance(r, dict) and "content" in r:
                content = r["content"]
            elif hasattr(r, "page_content"): # Fallback for standard LangChain Document classes
                content = r.page_content
            else:
                # Fallback to string conversion if the structure is atypical
                content = str(r)

            # CLEANUP LAYER: If the content itself is trapped inside a stringified list-dict string
            # e.g., [{'type': 'text', 'text': '...'}]
            if content.startswith("[") and "{'type':" in content:
                try:
                    # Clean up single quotes to valid JSON formatting representation if possible
                    json_valid = content.replace("'", '"')
                    parsed_list = json.loads(json_valid)
                    if isinstance(parsed_list, list) and len(parsed_list) > 0:
                        content = parsed_list[0].get("text", content)
                except Exception:
                    # If parsing drops, fallback to safe regex stripping to isolate the true core prose text
                    import re
                    match = re.search(r"'text':\s*'([^']*)'", content)
                    if match:
                        content = match.group(1)

            # Pull relevance scores safely
            score = getattr(r, "score", 0.0)
            if not score and isinstance(r, dict):
                score = r.get("score", 0.0)

            formatted_chunks.append(f"--- Document Chunk (Relevance: {score:.2f}) ---\n{content.strip()}")
        
        # Join with clear, distinct thematic structural spacing breaks
        return "\n\n---\n\n".join(formatted_chunks)