"""LangGraph orchestration for document-first chatbot with safe web fallback."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.core.config import Settings
from backend.core.database import DatabaseManager
from backend.engine.retrieval_service import RetrievalService


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    local_documents: list[str]
    cached_answer: str | None
    retrieved_context: str
    has_answer_in_docs: bool
    waiting_for_user_approval: bool
    user_approved_web_search: bool
    web_context: str
    final_answer: str
    answer_body: str
    thought_logs: list[str]


class IndusGuardianGraph:
    """Document-first chatbot with controlled web fallback."""

    def __init__(
        self,
        settings: Settings,
        db_manager: DatabaseManager,
        retrieval_service: RetrievalService,
    ) -> None:
        self.settings = settings
        self.db_manager = db_manager
        self.retrieval_service = retrieval_service

        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
            )

        self.graph = self._build()

    # ---------------- GRAPH ----------------
    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("ingest", self.ingest)
        workflow.add_node("cache", self.cache_lookup)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("check_answer", self.check_answer)
        workflow.add_node("ask_user", self.ask_user)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("ingest")

        workflow.add_edge("ingest", "cache")
        workflow.add_edge("cache", "retrieve")
        workflow.add_edge("retrieve", "check_answer")

        workflow.add_conditional_edges(
            "check_answer",
            self.route_after_check,
            {
                "generate": "generate",
                "ask_user": "ask_user",
            },
        )

        workflow.add_conditional_edges(
            "ask_user",
            self.route_after_user,
            {
                "web_search": "web_search",
                "end": END,
                "wait": END,
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ---------------- STEPS ----------------

    def ingest(self, state: AgentState) -> AgentState:
        return {**state, "thought_logs": ["Processing query..."]}

    def cache_lookup(self, state: AgentState) -> AgentState:
        cached = self.db_manager.get_cached_answer(
            state["question"],
            ttl_seconds=self.settings.semantic_cache_ttl_seconds,
        )
        return {**state, "cached_answer": cached}

    def retrieve(self, state: AgentState) -> AgentState:
        docs = state.get("local_documents", [])

        results = self.retrieval_service.retrieve_hybrid(
            query=state["question"],
            documents=docs,
            top_k=self.settings.top_k_retrieval,
        )

        context = self.retrieval_service.format_context(results)

        return {
            **state,
            "retrieved_context": context,
        }

    # ---------------- IMPORTANT FIX ----------------
    def check_answer(self, state: AgentState) -> AgentState:
        """
        STRICT RULE:
        Only accept answer if real document context exists.
        """

        context = state.get("retrieved_context", "").strip()

        has_answer = len(context) > 50  # threshold instead of weak boolean

        return {
            **state,
            "has_answer_in_docs": has_answer,
        }

    def route_after_check(self, state: AgentState) -> str:
        return "generate" if state.get("has_answer_in_docs") else "ask_user"

    # ---------------- ASK USER (NO HALLUCINATION) ----------------
    def ask_user(self, state: AgentState) -> AgentState:
        return {
            **state,
            "waiting_for_user_approval": True,
            "answer_body": (
                "I could not find an answer in your uploaded documents.\n\n"
                "Do you want me to search the internet for this information?"
            ),
        }

    def route_after_user(self, state: AgentState) -> str:
        if state.get("user_approved_web_search") is True:
            return "web_search"
        if state.get("user_approved_web_search") is False:
            return "end"
        return "wait"

    # ---------------- WEB SEARCH ----------------
    def web_search(self, state: AgentState) -> AgentState:
        if not self.settings.tavily_api_key:
            return {
                **state,
                "web_context": "Web search not available.",
            }

        tool = TavilySearchResults(
            api_key=self.settings.tavily_api_key,
            max_results=3,
        )

        result = tool.invoke({"query": state["question"]})

        return {
            **state,
            "web_context": str(result),
        }

    # ---------------- GENERATION ----------------
    def generate(self, state: AgentState) -> AgentState:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context = state.get("retrieved_context") or state.get("web_context")

        if not context:
            answer = "No information found in documents or web sources."
        else:
            prompt = f"""
You are a strict document-based assistant.

RULES:
- ONLY use provided context
- If context is weak, say you cannot find answer
- Do NOT hallucinate

Question:
{state['question']}

Context:
{context}

Answer:
"""
            answer = self.llm.invoke(prompt).content

        return {
            **state,
            "final_answer": f"[{now}] {answer}",
            "answer_body": answer,
        }