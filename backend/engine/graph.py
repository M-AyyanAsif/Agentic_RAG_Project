from __future__ import annotations
import os
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
        
        if settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",  
            google_api_key=settings.google_api_key,
            temperature=0.0,
            max_retries=3,      
            timeout=60.0,       
        )
        self.memory = MemorySaver()
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("ask_user", self.ask_user)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieve,
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
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["ask_user"] 
        )

    def retrieve(self, state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        active_session_id = configurable.get("thread_id") or state.get("session_id")
        
        question = state.get("question", "")
        generation_context = ""
        grading_context = ""
        target_namespace = "global_knowledge_base"

        logger.info(f"Graph Node Retrieval: Querying Knowledge Base namespace '{target_namespace}'")
        
        # Pull diverse context via the newly updated retriever
        results = self.retrieval_service.retrieve_hybrid(
            query=question, 
            session_id=target_namespace, 
            top_k=20
        )
        
        if results:
            try:
                generation_context = self.retrieval_service.format_context(results)
                grading_context = generation_context
            except Exception:
                grading_context = ""
                generation_context = ""
        
        def evaluate_relevance(context_text: str) -> bool:
            if not context_text.strip() or len(context_text.strip()) < 20:
                return False
            
            truncated_context = context_text[:20000]
            
            # CRITICAL FIX: The Grader prompt was previously failing because it was overwhelmed by mixed documents.
            grader_prompt = (
                "You are an elite Quality Assurance AI evaluating retrieved document chunks.\n"
                f"User Question: {question}\n\n"
                f"Retrieved Document Context:\n{truncated_context}\n\n"
                "TASK: Determine if the context contains ANY information relevant to answering the question.\n"
                "CRITICAL INSTRUCTION: Even if 90% of the context is irrelevant noise from other files, if there is a single paragraph or sentence that answers or partially addresses the question, you MUST output YES.\n"
                "Output ONLY 'YES' if relevant info exists, or 'NO' if the text is entirely unrelated."
            )
            try:
                grade_response = self.llm.invoke(grader_prompt)
                content = getattr(grade_response, 'content', grade_response)
                
                raw_text = ""
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            raw_text += " " + str(item.get('TEXT', item.get('text', str(item))))
                        else:
                            raw_text += " " + str(item)
                elif isinstance(content, dict):
                    raw_text = str(content.get('TEXT', content.get('text', str(content))))
                else:
                    raw_text = str(content)

                decision = raw_text.upper().strip()
                
                # Removing periods or markdown if generated
                if "YES" in decision:
                    logger.info("LLM Grader Decision: YES")
                    return True
                elif "NO" in decision:
                    logger.info("LLM Grader Decision: NO")
                    return False
                
                return False 
            except Exception as e:
                logger.error(f"Grader error safely caught: {e}")
                return False 

        has_info = evaluate_relevance(grading_context)
        
        if not has_info:
            logger.info("Vector search results failed validation. Context is insufficient for generation.")
        else:
            logger.info("Context validation successful. Documents match the user query.")

        return {
            "retrieved_context": generation_context if has_info else "",
            "has_answer_in_docs": has_info,
            "session_id": active_session_id,
            "user_approved_web_search": None,
            "web_context": "",
            "final_answer": "",
            "answer_body": "",
            "thought_logs": ["Evaluated document relevance against local vector metadata stores."]
        }

    def route_after_retrieve(self, state: AgentState):
        if state.get("has_answer_in_docs") is True:
            return "generate"
        return "ask_user"

    def ask_user(self, state: AgentState):
        return {
            "thought_logs": ["Execution paused. Document context insufficient. Awaiting human-in-the-loop web search approval..."]
        }

    def route_after_user(self, state: AgentState):
        if state.get("user_approved_web_search") is True:
            return "web_search"
        return "end"

    def web_search(self, state: AgentState):
        logger.info("Entering Web Search node via Tavily Engine...")
        if self.settings.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.settings.tavily_api_key

        try:
            tool = TavilySearchResults(tavily_api_key=self.settings.tavily_api_key, max_results=3)
            result = tool.invoke({"query": state.get("question", "")})
            
            if isinstance(result, list):
                clean_lines = []
                for idx, res in enumerate(result, 1):
                    url = res.get("url", "N/A")
                    content = res.get("content", "").strip()
                    clean_lines.append(f"[Web Reference #{idx}]\nSource URL: {url}\nExtract: {content}")
                web_ctx = "\n\n".join(clean_lines)
            else:
                web_ctx = str(result)
                
            logger.info("Successfully fetched and structured search results from Tavily.")
        except Exception as e:
            logger.error(f"Tavily lookup failed execution directly: {e}")
            web_ctx = f"Search failed error payload tracking: {str(e)}"

        return {
            "web_context": web_ctx,
            "thought_logs": ["Web query dispatch processed via Tavily pipelines."]
        }

    def generate(self, state: AgentState):
        is_web_search = state.get("user_approved_web_search") is True
        
        if is_web_search:
            context = state.get("web_context", "")
            source_type = "Web Search Results"
        else:
            context = state.get("retrieved_context", "")
            source_type = "Internal Knowledge Base"

        prompt = (
            f"You are Indus-Guardian AI, an advanced technical specialist system.\n"
            f"Answer the query using ONLY the provided {source_type} below.\n\n"
            f"CRITICAL FORMATTING RULES FOR BEAUTIFUL OUTPUT:\n"
            f"- Start with a clear, direct, and concise summary.\n"
            f"- Organize the details using clean Markdown headers (##, ###).\n"
            f"- Break down complex points into highly readable bulleted lists.\n"
            f"- Use bolding (**word**) to highlight critical technical terminology or metrics.\n"
            f"- If the answer spans multiple concepts, structure it logically.\n"
            f"- Do not include unnecessary conversational filler.\n\n"
            f"Context Data ({source_type}):\n{context}\n\n"
            f"Question: {state.get('question', '')}\n"
            f"Answer:"
        )
        
        try:
            logger.info("Forwarding context to LLM for final beautiful generation...")
            response = self.llm.invoke(prompt)
            generated_text = str(response.content) if hasattr(response, 'content') else str(response)
            
            try:
                self.db_manager.upsert_cached_answer(str(state.get("question", "")), generated_text)
                logger.info("Cache entry synchronization completed successfully.")
            except Exception as cache_err:
                logger.warning(f"Cache save bypassed: {cache_err}")

        except Exception as api_error:
            logger.error(f"Upstream LLM Generation Failed. Error: {api_error}")
            generated_text = (
                "## ⚠️ API Service Congestion\n\n"
                "The Google Gemini API is currently experiencing a temporary slowdown. Please resubmit your query."
            )
            
        return {
            "final_answer": generated_text,
            "answer_body": generated_text
        }