"""
LangGraph orchestration with BREAKPOINTS for safe web fallback.
- Production Optimized: Correctly forwards thread_id tracking to retrieval engines.
- State-Locked: Precludes namespace drop-offs or null-state overwrites.
"""

from __future__ import annotations
from datetime import datetime
from typing import TypedDict, Annotated, Any, Dict
import operator
import logging

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.runnables import RunnableConfig

from backend.core.config import Settings
from backend.core.database import DatabaseManager

logger = logging.getLogger("indus-guardian.graph")

class AgentState(TypedDict):
    question: str
    session_id: str | None
    retrieved_context: str
    has_answer_in_docs: bool
    user_approved_web_search: bool | None 
    web_context: str
    final_answer: str
    answer_body: str
    thought_logs: Annotated[list[str], operator.add] 

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
        self.memory = MemorySaver()
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        # Registering nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("ask_user", self.ask_user)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve")

        # 1. After retrieval evaluation, route based on text availability
        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieve,
            {
                "generate": "generate",
                "ask_user": "ask_user",
            },
        )

        # 2. State breakpoint resolution rules
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

        # Enforces human-in-the-loop state break points right before asking user
        return workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["ask_user"] 
        )

    # ---------------- NODES ----------------

    def retrieve(self, state: AgentState, config: RunnableConfig):
        """Exhaustive document retrieval node leveraging context threading."""
        # SENIOR MOVE: Safely pull the active thread session context out of the LangGraph thread config
        configurable = config.get("configurable", {})
        active_session_id = configurable.get("thread_id") or state.get("session_id")
        
        question = state["question"]
        context = ""

        if active_session_id and active_session_id != "None":
            logger.info(f"Graph Node Retrieval: Processing queries against namespace '{active_session_id}'")
            
            # 1. Primary Vector Search
            results = self.retrieval_service.retrieve_hybrid(
                query=question, 
                session_id=active_session_id, 
                top_k=7
            )
            if results:
                context = self.retrieval_service.format_context(results)
            
            # 2. SQLite Context Backup Layer 
            if not context or len(context.strip()) < 20:
                logger.info("Graph Vector search narrow. Executing relational database extraction backup...")
                fallback_docs = self.db_manager.get_documents(active_session_id)
                if fallback_docs:
                    context = "\n".join([doc[0] for doc in fallback_docs])
        else:
            logger.warning("Graph execution running without thread configuration contexts. Bypassing document indexing.")

        # Evaluate if we compiled valid structural context
        has_info = len(context.strip()) > 50 
        
        return {
            "retrieved_context": context,
            "has_answer_in_docs": has_info,
            "session_id": active_session_id,
            "thought_logs": ["Analyzed localized system knowledge structures."]
        }

    def route_after_retrieve(self, state: AgentState):
        if state.get("has_answer_in_docs") is True:
            return "generate"
        return "ask_user"

    def ask_user(self, state: AgentState):
        return {
            "thought_logs": ["Execution paused. Evaluating web lookup privileges..."]
        }

    def route_after_user(self, state: AgentState):
        if state.get("user_approved_web_search") is True:
            return "web_search"
        return "end"

    def web_search(self, state: AgentState):
        tool = TavilySearchResults(api_key=self.settings.tavily_api_key, max_results=3)
        result = tool.invoke({"query": state["question"]})
        return {
            "web_context": str(result),
            "thought_logs": ["Web query dispatch processed via Tavily pipelines."]
        }

    def generate(self, state: AgentState):
        context = state.get("retrieved_context")
        if not context or len(context.strip()) < 20:
            context = state.get("web_context", "")

        # SENIOR PROMPT DESIGN: Explicitly mandate scannable structural layouts
        prompt = (
            f"You are Indus-Guardian AI, an advanced technical specialist system.\n"
            f"Answer the query using the retrieved context block details below.\n\n"
            f"CRITICAL FORMATTING RULES:\n"
            f"- Organize your response using clean Markdown headers (##, ###).\n"
            f"- Break down complex points into clear, bulleted or numbered lists.\n"
            f"- Use bolding (**word**) to highlight critical technical metrics or actions.\n"
            f"- Ensure there is a blank line between paragraphs to preserve clean formatting.\n"
            f"- Keep sentences clear, direct, and highly professional.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n"
            f"Answer:"
        )
        
        response = self.llm.invoke(prompt)
        
        generated_text = str(response.content) if hasattr(response, 'content') else str(response)
            
        try:
            self.db_manager.upsert_cached_answer(str(state["question"]), generated_text)
        except Exception as e:
            logger.warning(f"Cache save bypassed: {e}")
            
        return {
            "final_answer": generated_text,
            "answer_body": generated_text
        }