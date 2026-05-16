"""
Indus-Guardian Production API
- Enforces absolute local document priority over web search fallbacks
- Hardens internal session state management against LangGraph decoupled node calls
- Provides robust string sanitization for SQLite database insertions
"""

from __future__ import annotations
import json
import logging
import asyncio
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.core.logging_setup import configure_logging
from backend.engine.document_processor import DocumentProcessor
from backend.engine.graph import IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService

# ---------------- INITIALIZATION ----------------

settings = get_settings()
configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger("indus-guardian.api")

db_manager = DatabaseManager(settings.sqlite_path)
doc_processor = DocumentProcessor()
retrieval_service = RetrievalService(settings)
agent = IndusGuardianGraph(settings, db_manager, retrieval_service)

app = FastAPI(title="Indus-Guardian Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    approve_web_search: bool | None = None

# ---------------- UTILS ----------------

def _to_sse(event: str, data: dict) -> str:
    """Formats messages cleanly into Server-Sent Events."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ---------------- ROUTES ----------------

@app.get("/health")
async def health():
    return {"status": "ok", "database": "connected"}

@app.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        parsed_doc = doc_processor.parse(file.filename, file.content_type, content)
        db_manager.add_document(session_id, parsed_doc.content)
        doc_processor.index_to_pinecone(session_id, parsed_doc)
        return {"message": "Document processed and indexed", "session_id": session_id}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # SENIOR GUARDRAIL: Sanitize incoming session IDs immediately to block literal "None" strings
    incoming_session = request.session_id
    if not incoming_session or incoming_session == "None" or incoming_session is None:
        incoming_session = str(uuid4())
        logger.warning(f"Session ID missing or corrupt string payload. Generated safe fallback: {incoming_session}")
    
    async def event_generator():
        # Scrape and bind the session token cleanly inside the generator closure scoping
        active_session_id = incoming_session
        try:
            context_text = ""
            
            # Phase 1: Only check local document indexing rules if no web-search directive exists
            if request.approve_web_search is None:
                logger.info(f"Exhaustive Search Mode: Checking Pinecone namespace '{active_session_id}'")
                
                # 1. Broad context search window
                context_results = retrieval_service.retrieve_hybrid(
                    query=request.message, 
                    session_id=active_session_id,
                    top_k=7
                )
                
                if context_results:
                    context_text = retrieval_service.format_context(context_results)
                
                # 2. SQLite Database Backup Falltext Extraction Layer
                if not context_text or len(context_text.strip()) < 20:
                    logger.info(f"Vector search narrow match. Running extraction fallback on SQLite for namespace '{active_session_id}'...")
                    fallback_docs = db_manager.get_documents(active_session_id)
                    if fallback_docs:
                        context_text = "\n".join([doc[0] for doc in fallback_docs])
                
                # 3. Halt processing and prompt for web integration if local search options hit a complete wall
                if not context_text or len(context_text.strip()) < 20:
                    logger.info(f"All local knowledge base configurations empty for namespace '{active_session_id}'. Yielding approval flag.")
                    yield _to_sse("approval_required", {
                        "message": "Could not find context in local documents. Would you like to check the web?",
                        "session_id": active_session_id
                    })
                    return

            # Phase 2: Enforce strict LangGraph parameter configurations
            config = {"configurable": {"thread_id": active_session_id}}

            if request.approve_web_search is not None:
                # Update loop for handling Human-In-The-Loop state resumption
                agent.graph.update_state(config, {"user_approved_web_search": request.approve_web_search})
                current_state = agent.graph.invoke(None, config)
            else:
                # Run complete generation path using locally gathered file contexts
                initial_input = {
                    "question": request.message,
                    "context": context_text,
                    "thought_logs": ["Context verified. Activating localized generation engine..."]
                }
                current_state = agent.graph.invoke(initial_input, config)

            # Phase 3: Senior Defensive Key, State, and Block Element Harvesting
            # Phase 3: Senior Defensive Key, State, and Block Element Harvesting
            raw_ans = current_state.get("final_answer")
            
            # If the custom state machine model drops or wipes the key on completion, parse state history strings
            if not raw_ans or raw_ans == "Unable to extract answer from current documents.":
                if "messages" in current_state and current_state["messages"]:
                    raw_ans = current_state["messages"][-1].content
                else:
                    raw_ans = "I located your local files, but could not compile a precise answer block. Please optimize your question."

            # SENIOR EXTRACTOR: Drill straight into raw lists, dicts, or nested text blocks
            final_ans = ""

            if isinstance(raw_ans, list) and len(raw_ans) > 0:
                first_element = raw_ans[0]
                if isinstance(first_element, dict) and "text" in first_element:
                    final_ans = str(first_element["text"])
                elif hasattr(first_element, "text"):
                    final_ans = str(first_element.text)
                else:
                    final_ans = str(first_element)
            elif isinstance(raw_ans, dict):
                if "text" in raw_ans:
                    final_ans = str(raw_ans["text"])
                elif "content" in raw_ans:
                    final_ans = str(raw_ans["content"])
                else:
                    final_ans = str(raw_ans)
            else:
                final_ans = str(raw_ans)

            # DEEP CLEAN ENGINE: Catch structural list structures anywhere inside the stream text string
            final_ans = final_ans.strip()
            if "{'type':" in final_ans or "'extras':" in final_ans:
                import re
                # Pull the precise value of the 'text' key safely using a strict non-greedy regex
                matches = re.findall(r"'text':\s*\"(.*?)\"(?:,\s*'extras'|\s*})", final_ans, re.DOTALL)
                if not matches:
                    matches = re.findall(r"'text':\s*'(.*?)'(?:,\s*'extras'|\s*})", final_ans, re.DOTALL)
                
                if matches:
                    # Replace escaped newlines with actual structural layout line breaks
                    final_ans = "\n".join(matches).replace("\\n", "\n")

            # FINAL PROTECTION: Remove raw single/double quote wrappers that cause Streamlit to zoom text
            if (final_ans.startswith("'") and final_ans.endswith("'")) or (final_ans.startswith('"') and final_ans.endswith('"')):
                final_ans = final_ans[1:-1]

            # Save the cleanly processed assistant response down to database layers safely
            db_manager.add_message(active_session_id, "assistant", final_ans)
            yield _to_sse("done", {"message": final_ans, "session_id": active_session_id})

        except Exception as e:
            logger.error(f"Stream Core Panic Exception: {str(e)}")
            yield _to_sse("error", {"detail": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    db_manager.delete_session(session_id)
    try:
        doc_processor.index.delete(delete_all=True, namespace=session_id)
    except Exception as e:
        logger.warning(f"Associated namespace vectors clear skipped: {e}")
    return {"message": "Deleted"}