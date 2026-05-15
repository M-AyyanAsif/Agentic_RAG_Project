"""Tests for graph routing decisions (Synced with Indus-Guardian Graph)."""

from __future__ import annotations
import pytest
from backend.core.config import Settings
from backend.core.database import DatabaseManager
from backend.engine.graph import IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService

@pytest.fixture(scope="module")
def graph():
    """Build the graph once for all routing tests."""
    settings = Settings(
        sqlite_path=":memory:", # Use RAM for testing speed
        semantic_cache_ttl_seconds=1,
    )
    db = DatabaseManager(settings.sqlite_path)
    retrieval = RetrievalService(settings)
    return IndusGuardianGraph(settings, db, retrieval)

# ---------------- ROUTING TESTS ----------------

def test_route_generate_when_docs_found(graph: IndusGuardianGraph) -> None:
    """If docs have the answer, go to generation."""
    state = {"has_answer_in_docs": True}
    # Matches the method name in our refactored graph.py
    result = graph.route_after_retrieve(state) 
    assert result == "generate"

def test_route_ask_user_when_docs_missing(graph: IndusGuardianGraph) -> None:
    """If docs are empty, trigger the 'ask_user' interrupt."""
    state = {"has_answer_in_docs": False}
    result = graph.route_after_retrieve(state)
    assert result == "ask_user"

def test_route_web_search_on_approval(graph: IndusGuardianGraph) -> None:
    """If the user clicks 'Yes', proceed to Tavily."""
    state = {"user_approved_web_search": True}
    result = graph.route_after_user(state)
    assert result == "web_search"

def test_route_end_on_disapproval(graph: IndusGuardianGraph) -> None:
    """If the user clicks 'No', stop the agent."""
    state = {"user_approved_web_search": False}
    result = graph.route_after_user(state)
    assert result == "end"