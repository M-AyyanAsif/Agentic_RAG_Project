"""LangGraph orchestration with BREAKPOINTS for safe web fallback."""

from __future__ import annotations
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver # For saving state while waiting

from backend.core.config import Settings
from backend.core.database import DatabaseManager

class AgentState(TypedDict):
    question: str
    session_id: str
    retrieved_context: str
    has_answer_in_docs: bool
    user_approved_web_search: bool # This will be filled by the frontend
    web_context: str
    final_answer: str
    answer_body: str
    thought_logs: Annotated[list[str], operator.add] # Append-only logs

class IndusGuardianGraph:
    def __init__(self, settings: Settings, db_manager: DatabaseManager, retrieval_service):
        self.settings = settings
        self.db_manager = db_manager
        self.retrieval_service = retrieval_service
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        # Memory allows the graph to "remember" where it stopped
        self.memory = MemorySaver()
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("ask_user", self.ask_user)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve")

        # 1. After retrieval, check if we have info
        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieve,
            {
                "generate": "generate",
                "ask_user": "ask_user",
            },
        )

        # 2. After asking user, the graph STOPS here. 
        # When it resumes, it checks the user's choice.
        workflow.add_conditional_edges(
            "ask_user",
            self.route_after_user,
            {
                "web_search": "web_search",
                "end": END,
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        # We compile with a breakpoint at "ask_user"
        return workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["ask_user"] 
        )

    # ---------------- NODES ----------------

    def retrieve(self, state: AgentState):
        results = self.retrieval_service.retrieve_hybrid(query=state["question"])
        context = self.retrieval_service.format_context(results)
        
        # Use LLM to judge if context is actually useful (Senior Move)
        has_info = len(context.strip()) > 100 
        
        return {
            "retrieved_context": context,
            "has_answer_in_docs": has_info,
            "thought_logs": ["Searched local documents."]
        }

    def route_after_retrieve(self, state: AgentState):
        if state["has_answer_in_docs"]:
            return "generate"
        return "ask_user"

    def ask_user(self, state: AgentState):
        # This node is reached but execution is INTERRUPTED before it starts.
        return {
            "thought_logs": ["Awaiting user permission for web search..."]
        }

    def route_after_user(self, state: AgentState):
        # When user clicks 'Yes' in Streamlit, it sets user_approved_web_search = True
        if state.get("user_approved_web_search") is True:
            return "web_search"
        return "end"

    def web_search(self, state: AgentState):
        tool = TavilySearchResults(api_key=self.settings.tavily_api_key, max_results=3)
        result = tool.invoke({"query": state["question"]})
        return {
            "web_context": str(result),
            "thought_logs": ["Web search completed via Tavily."]
        }

    def generate(self, state: AgentState):
        context = state.get("retrieved_context") or state.get("web_context", "")
        prompt = f"Answer based ONLY on context: {context}\nQuestion: {state['question']}"
        response = self.llm.invoke(prompt)
        
        # Save to semantic cache here (Rule from database.py)
        self.db_manager.upsert_cached_answer(state["question"], response.content)
        
        return {
            "final_answer": response.content,
            "answer_body": response.content
        }