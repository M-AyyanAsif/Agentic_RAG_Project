"""Hybrid retrieval service (safe RAG version: no hallucination support)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha1
from typing import Sequence

from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    source: str
    content: str
    score: float


class RetrievalService:
    """
    STRICT retrieval system:
    - returns only relevant document chunks
    - filters weak matches
    - supports safe RAG (no hallucination)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reranker: CrossEncoder | None = None
        self._embedder: SentenceTransformer | None = None

        self._pinecone = (
            Pinecone(api_key=settings.pinecone_api_key)
            if settings.pinecone_api_key
            else None
        )

        self._index = (
            self._pinecone.Index(settings.pinecone_index_name)
            if self._pinecone
            else None
        )

    # ---------------- MODELS ----------------
    @property
    def reranker(self) -> CrossEncoder | None:
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            except Exception as exc:
                logger.warning("Reranker unavailable: %s", exc)
                self._reranker = None
        return self._reranker

    @property
    def embedder(self) -> SentenceTransformer | None:
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(
                    self.settings.embedding_model_name
                )
            except Exception as exc:
                logger.warning("Embedder unavailable: %s", exc)
                self._embedder = None
        return self._embedder

    # ---------------- CHUNKING ----------------
    @staticmethod
    def _chunk_documents(documents: list[str], chunk_size: int = 800) -> list[str]:
        chunks = []

        for doc in documents:
            if len(doc) <= chunk_size:
                chunks.append(doc)
                continue

            for i in range(0, len(doc), chunk_size):
                chunks.append(doc[i:i + chunk_size])

        return chunks

    # ---------------- MAIN RETRIEVAL ----------------
    def retrieve_hybrid(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RetrievalResult]:

        chunks = self._chunk_documents(documents)

        if not chunks:
            return []

        pinecone_results = self._pinecone_search(query, chunks, top_k)
        bm25_results = self._bm25_search(query, chunks, top_k)

        merged = pinecone_results + bm25_results

        # sort by score
        merged.sort(key=lambda x: x.score, reverse=True)

        # STRICT FILTERING (IMPORTANT)
        filtered = [r for r in merged if r.score >= 0.2]

        if not filtered:
            return []

        return self._rerank(query, filtered, self.settings.top_k_after_rerank)

    # ---------------- PINECONE ----------------
    def _pinecone_search(
        self, query: str, documents: list[str], top_k: int
    ) -> list[RetrievalResult]:

        if not self._index or not self.embedder:
            return []

        embeddings = self.embedder.encode(documents[:100], normalize_embeddings=True)

        vectors = []
        for text, emb in zip(documents[:100], embeddings):
            vectors.append(
                {
                    "id": sha1(text.encode()).hexdigest(),
                    "values": emb.tolist(),
                    "metadata": {"text": text},
                }
            )

        # ⚠ REMOVED continuous upsert per query (bug fix)
        # self._index.upsert(vectors=vectors)

        query_vector = self.embedder.encode([query], normalize_embeddings=True)[0]

        res = self._index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
        )

        matches = res.get("matches", [])

        return [
            RetrievalResult(
                source="pinecone",
                content=m["metadata"]["text"],
                score=float(m.get("score", 0.0)),
            )
            for m in matches
            if m.get("score", 0) > 0.2
        ]

    # ---------------- BM25 ----------------
    def _bm25_search(
        self, query: str, documents: list[str], top_k: int
    ) -> list[RetrievalResult]:

        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)

        scores = bm25.get_scores(query.lower().split())

        results = [
            RetrievalResult("bm25", doc, float(score))
            for doc, score in zip(documents, scores)
            if score > 0.2
        ]

        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

    # ---------------- RERANK ----------------
    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:

        if not candidates:
            return []

        if not self.reranker:
            return candidates[:top_k]

        pairs = [[query, c.content] for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RetrievalResult(
                source=c.source,
                content=c.content,
                score=float(score),
            )
            for c, score in ranked[:top_k]
        ]

    # ---------------- FORMAT ----------------
    def format_context(self, results: list[RetrievalResult]) -> str:
        if not results:
            return ""

        return "\n".join(
            f"[{i+1} | {r.source} | {r.score:.2f}] {r.content}"
            for i, r in enumerate(results)
        )