"""Hybrid retrieval service (Optimized for Google Cloud Embeddings)."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List
import google.generativeai as genai
from pinecone import Pinecone

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
            result = genai.embed_content(
                model=self.settings.embedding_model_name, # Usually models/text-embedding-004
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Google Embedding failed: {e}")
            return []

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Queries Pinecone using the Google-generated vector.
        """
        if not self._index:
            logger.warning("Pinecone Index not initialized. Skipping retrieval.")
            return []

        # 1. Get the vector for the user's question
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []

        # 2. Query Pinecone
        res = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = res.get("matches", [])
        
        # 3. Filter and Format (Score > 0.4 is safe for Google Embeddings)
        results = [
            RetrievalResult(
                source="pinecone",
                content=m["metadata"].get("text", ""),
                score=float(m.get("score", 0.0)),
            )
            for m in matches if m.get("score", 0) > 0.4
        ]

        logger.info(f"Retrieved {len(results)} relevant chunks from Pinecone.")
        return results

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Converts results into a single string for the LLM prompt."""
        if not results:
            return ""
        
        return "\n\n".join(
            f"--- Document Chunk ({r.score:.2f}) ---\n{r.content}" 
            for r in results
        )