"""Tests for graph routing decisions (improved version)."""

from __future__ import annotations

from backend.core.config import Settings
from backend.core.database import DatabaseManager
from backend.engine.graph import IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService


def _build_graph() -> IndusGuardianGraph:
    settings = Settings(
        sqlite_path="backend/data/test_graph.db",
        semantic_cache_ttl_seconds=1,
    )
    db = DatabaseManager(settings.sqlite_path)
    retrieval = RetrievalService(settings)
    return IndusGuardianGraph(settings, db, retrieval)


# ---------------- ROUTING TESTS ----------------

def test_route_generate_when_relevant() -> None:
    graph = _build_graph()

    state = {"relevant": True}
    result = graph.route_after_grade(state)

    assert result == "generate"


def test_route_ask_user_when_not_relevant() -> None:
    graph = _build_graph()

    state = {"relevant": False}
    result = graph.route_after_grade(state)

    assert result == "user_approval_node"


# ---------------- EXTRA SAFETY TEST ----------------

def test_no_invalid_route_values() -> None:
    graph = _build_graph()

    result = graph.route_after_grade({"relevant": True})

    assert result in {"generate", "user_approval_node"}