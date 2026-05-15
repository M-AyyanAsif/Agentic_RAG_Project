"""FastAPI entrypoint for Indus-Guardian (production-safe version)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.core.logging_setup import configure_logging
from backend.engine.document_processor import DocumentProcessor
from backend.engine.graph import AgentState, IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService

# ---------------- INIT ----------------

settings = get_settings()
configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger("indus-guardian.api")

db_manager = DatabaseManager(settings.sqlite_path)
document_processor = DocumentProcessor()
retrieval_service = RetrievalService(settings)
agent_graph = IndusGuardianGraph(settings, db_manager, retrieval_service)

Path(settings.backup_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.app_name)

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


class DeleteSessionRequest(BaseModel):
    session_id: str


# ---------------- ROUTES ----------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()

    if len(content) > settings.max_uploaded_file_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    if not file.filename or not file.content_type:
        raise HTTPException(status_code=400, detail="Invalid upload.")

    parsed = document_processor.parse(file.filename, file.content_type, content)

    if not parsed.content.strip():
        raise HTTPException(status_code=400, detail="No readable text found.")

    db_manager.add_document(session_id, parsed.content)

    return JSONResponse(
        {
            "message": f"Uploaded {parsed.filename}",
            "chars_indexed": len(parsed.content),
            "session_id": session_id,
        }
    )


# ---------------- STREAM HELP ----------------

def _to_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# ---------------- CHAT STREAM ----------------

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    session_id = request.session_id or str(uuid4())
    question = request.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Empty message.")

    db_manager.add_message(session_id, "user", question)

    documents = db_manager.get_documents(session_id)

    state: AgentState = {
        "question": question,
        "session_id": session_id,
        "local_documents": documents,
        "user_approved_web_search": request.approve_web_search,
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        # ❗ SAFETY: No documents = force approval flow
        if not documents:
            yield _to_sse(
                "approval_required",
                {
                    "session_id": session_id,
                    "prompt": "No documents found. Do you want to search the web?",
                },
            )
            return

        try:
            result_state = agent_graph.graph.invoke(state)
        except Exception as exc:
            logger.exception("Graph execution failed: %s", exc)
            yield _to_sse(
                "done",
                {
                    "message": (
                        "I could not process this query due to a backend configuration "
                        "issue. Please verify API keys and model settings."
                    )
                },
            )
            return

        for log_line in result_state.get("thought_logs", []):
            yield _to_sse("thought", {"log": log_line})

        # STRICT WEB APPROVAL LOGIC
        if result_state.get("waiting_for_user_approval"):
            if request.approve_web_search is None:
                yield _to_sse(
                    "approval_required",
                    {
                        "prompt": "Not found in documents. Allow web search?",
                    },
                )
                return
            elif request.approve_web_search is False:
                yield _to_sse(
                    "done",
                    {"message": "Conversation ended as per user choice."},
                )
                return

        answer = result_state.get("final_answer", "No answer generated.")

        db_manager.add_message(session_id, "assistant", answer)

        for token in answer.split():
            yield _to_sse("token", {"token": token + " "})

        yield _to_sse("done", {"session_id": session_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------- SESSIONS ----------------

@app.get("/sessions")
async def list_sessions() -> dict[str, list[str]]:
    return {"sessions": db_manager.list_sessions()}


@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "messages": [
    {
        "role": r.role, 
        "content": r.content, 
        "timestamp": str(r.created_at) if hasattr(r, "created_at") else ""
    } 
    for r in db_manager.get_messages(session_id)
],
    }


@app.delete("/sessions")
async def delete_session(request: DeleteSessionRequest) -> dict[str, str]:
    db_manager.delete_session(request.session_id)
    return {"message": f"Deleted session {request.session_id}"}