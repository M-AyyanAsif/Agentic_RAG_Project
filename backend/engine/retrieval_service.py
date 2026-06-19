"""Hybrid retrieval service (Optimized for Google Cloud Embeddings)."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Any
import json
import re
from google import genai
from google.genai import types
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
        
        # Initialize Modern Google GenAI Client
        self.client = genai.Client(api_key=settings.google_api_key)
        
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
        """Fetch vector from Google API using clean content parameters."""
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-2", 
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=1024
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Google Embedding failed: {e}")
            return []

    def retrieve_hybrid(
        self, 
        query: str, 
        session_id: Optional[str] = None, 
        top_k: int = 35,
        score_threshold: float = 0.35  # Calibrated for pure gemini-embedding-2 vector distances
    ) -> list[RetrievalResult]:
        """
        Queries Pinecone using clean, structural-noise-free vectors.
        Implements Source Diversity to prevent Vector Hijacking.
        """
        if not self._index:
            logger.warning("Pinecone Index not initialized. Skipping retrieval.")
            return []

        cleaned_query = re.sub(
            r'(?i)\b(from\s+doc(?:ument)?\s*\d+|according\s+to\s+doc(?:ument)?\s*\d+|in\s+doc(?:ument)?\s*\d+|doc(?:ument)?\s*\d+)\b', 
            '', 
            query
        )
        cleaned_query = re.sub(r'\s+,|\s+', ' ', cleaned_query).strip()
        
        if not cleaned_query:
            cleaned_query = query

        logger.info(f"Original RAG Query: '{query}' -> Formatted Semantic Match: '{cleaned_query}'")

        query_vector = self._get_embedding(cleaned_query)
        if not query_vector:
            return []

        try:
            res = self._index.query(
                vector=query_vector,
                top_k=top_k,  
                include_metadata=True,
                namespace=session_id 
            )
        except Exception as e:
            logger.error(f"Pinecone Query failed: {e}")
            return []

        matches = res.get("matches", [])
        
        results = []
        source_counts = {}
        MAX_CHUNKS_PER_SOURCE = 4  # CRITICAL FIX: Forces diversity, stops SQL doc from hijacking everything.

        for m in matches:
            score = float(m.get("score", 0.0))
            
            if score < score_threshold:
                continue
                
            metadata = m.get("metadata", {})
            
            # Identify the source document (Fallback to "unknown" if missing)
            doc_source = metadata.get("source", metadata.get("filename", "unknown_source"))
            
            # Diversity check: If we already have enough chunks from this document, skip to the next one
            if source_counts.get(doc_source, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
                
            source_counts[doc_source] = source_counts.get(doc_source, 0) + 1
            
            content = metadata.get("text") or metadata.get("content") or ""
            
            results.append(RetrievalResult(
                source=doc_source,
                content=content,
                score=score,
            ))

        logger.info(f"Retrieved {len(results)} diverse chunks from {len(source_counts)} unique sources.")
        return results

    def format_context(self, results: list[Any]) -> str:
        """Converts retrieval results into a clean string window for the LLM prompt."""
        if not results:
            return ""
        
        formatted_chunks = []
        for r in results:
            content = ""
            if hasattr(r, "content"):
                content = r.content
            elif isinstance(r, dict) and "content" in r:
                content = r["content"]
            elif hasattr(r, "page_content"): 
                content = r.page_content
            else:
                content = str(r)

            if isinstance(content, str):
                if content.startswith("[") and "{'type':" in content:
                    try:
                        json_valid = content.replace("'", '"')
                        parsed_list = json.loads(json_valid)
                        if isinstance(parsed_list, list) and len(parsed_list) > 0:
                            content = parsed_list[0].get("text", content)
                    except Exception:
                        match = re.search(r"'text':\s*'([^']*)'", content)
                        if match:
                            content = match.group(1)

            score = getattr(r, "score", 0.0)
            source = getattr(r, "source", "unknown_source")
            if not score and isinstance(r, dict):
                score = r.get("score", 0.0)
                source = r.get("source", "unknown_source")

            clean_text = str(content).strip()
            if clean_text:
                # Included source tag directly in the prompt context so the LLM knows where it came from
                formatted_chunks.append(f"--- Source: {source} | Match Score: {score:.3f} ---\n{clean_text}")
        
        return "\n\n".join(formatted_chunks)