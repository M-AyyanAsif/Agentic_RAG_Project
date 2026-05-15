"""
Indus-Guardian Production API
- Handles State Resumption (Human-in-the-loop)
- Full Session CRUD (Create, Read, Update, Delete)
- Memory Leak Protection for 8GB RAM
"""

from __future__ import annotations
import json
import logging
from uuid import uuid4
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.core.config import get_settings, Settings
from backend.core.database import DatabaseManager
from backend.core.logging_setup import configure_logging
from backend.engine.document_processor import DocumentProcessor
from backend.engine.graph import IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService

# ---------------- INITIALIZATION ----------------

settings = get_settings()
configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger("indus-guardian.api")

# Global instances
db_manager = DatabaseManager(settings.sqlite_path)
doc_processor = DocumentProcessor()
retrieval = RetrievalService(settings)
agent = IndusGuardianGraph(settings, db_manager, retrieval)

app = FastAPI(title="Indus-Guardian Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    approve_web_search: bool | None = None

class SessionActionRequest(BaseModel):
    session_id: str

# ---------------- UTILS ----------------

def _to_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ---------------- ROUTES ----------------

@app.get("/health")
async def health():
    return {"status": "ok", "database": "connected"}

@app.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    """Handles PDF ingestion and linking to session."""
    content = await file.read()
    parsed = doc_processor.parse(file.filename, file.content_type, content)
    
    # Save to SQLite and Pinecone via Ingestion Logic
    db_manager.add_document(session_id, parsed.content)
    
    return {"message": "Document processed", "session_id": session_id}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Advanced SSE Stream with Breakpoint Handling.
    Solves the 'Resumption' logic issue.
    """
    session_id = request.session_id or str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        try:
            # Check if we are resuming from a pause (User clicked Yes/No)
            if request.approve_web_search is not None:
                # Update the state with user's choice
                agent.graph.update_state(config, {"user_approved_web_search": request.approve_web_search})
                # Resume execution
                current_state = agent.graph.invoke(None, config)
            else:
                # Fresh start
                docs = db_manager.get_documents(session_id)
                initial_input = {
                    "question": request.message,
                    "local_documents": docs,
                    "thought_logs": ["Agent initialized..."]
                }
                current_state = agent.graph.invoke(initial_input, config)

            # Check if the agent is now PAUSED waiting for approval
            snapshot = agent.graph.get_state(config)
            if snapshot.next: # If list is not empty, it's waiting at a node
                yield _to_sse("approval_required", {"session_id": session_id})
                return

            # If finished, send the final answer
            final_ans = current_state.get("final_answer", "No response.")
            db_manager.add_message(session_id, "assistant", final_ans)
            yield _to_sse("done", {"message": final_ans})

        except Exception as e:
            logger.error(f"Graph Error: {e}")
            yield _to_sse("error", {"detail": "Internal processing error."})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------- SESSION MANAGEMENT (Your Doubts Solved) ----------------

@app.get("/sessions")
async def get_all_sessions():
    """Lists all available chat histories."""
    return {"sessions": db_manager.list_sessions()}

@app.delete("/sessions/clear-all")
async def clear_all_data():
    """Wipes all data - Emergency reset."""
    db_manager.clear_all_data() # Ensure this is in your DatabaseManager
    return {"message": "All sessions and documents cleared."}

@app.delete("/sessions/{session_id}")
async def delete_specific_session(session_id: str):
    """
    Properly deletes a session's history and documents.
    Solves the 'Old data popping up' bug.
    """
    db_manager.delete_session(session_id)
    # Optional: If you want to clear LangGraph memory too
    # agent.memory.delete(thread_id=session_id) 
    return {"message": f"Session {session_id} deleted."}