from __future__ import annotations
import json
import logging
import re
from typing import Any
from uuid import uuid4
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.core.logging_setup import configure_logging
from backend.engine.document_processor import DocumentProcessor
from backend.engine.graph import IndusGuardianGraph
from backend.engine.retrieval_service import RetrievalService

# Initialize
settings = get_settings()
configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger("indus-guardian.api")

db_manager = DatabaseManager(settings.sqlite_path)
doc_processor = DocumentProcessor()
retrieval_service = RetrievalService(settings)
agent = IndusGuardianGraph(settings, db_manager, retrieval_service)

app = FastAPI(title="Indus-Guardian Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    approve_web_search: bool | None = None

def _to_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def _sanitize_ai_response(raw_ans: Any) -> str:
    if not raw_ans:
        return ""

    if isinstance(raw_ans, AIMessage):
        raw_ans = raw_ans.content

    if isinstance(raw_ans, list):
        if len(raw_ans) > 0:
            return _sanitize_ai_response(raw_ans[0])
        return ""

    if isinstance(raw_ans, dict):
        for key in ["text", "content", "message", "final_answer"]:
            if key in raw_ans and raw_ans[key]:
                return _sanitize_ai_response(raw_ans[key])
        
        if raw_ans.get("type") == "text":
            for k, v in raw_ans.items():
                if k != "type" and isinstance(v, str) and len(v) > 5:
                    return v.strip()
        
        for v in raw_ans.values():
            if isinstance(v, str) and len(v) > 5:
                return v.strip()

    if isinstance(raw_ans, str):
        cleaned = raw_ans.strip()
        
        if (cleaned.startswith("[") and cleaned.endswith("]")) or (cleaned.startswith("{") and cleaned.endswith("}")):
            try:
                parsed = json.loads(cleaned)
                return _sanitize_ai_response(parsed)
            except Exception:
                pass
                
        cleaned = re.sub(r'^[\s]*\{.*?"text":\s*"', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'"\s*\}[\s]*$', '', cleaned, flags=re.DOTALL)
        
        cleaned = cleaned.replace("\\n", "\n")
        return cleaned.strip()

    return str(raw_ans).strip()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # FIXED: Map the execution thread directly to the persistent session ID.
    # Appending random UUID identifiers on every API invocation broke checkpointing memory entirely.
    base_session = request.session_id or str(uuid4())
    execution_thread_id = base_session 
    
    if request.approve_web_search is None:
        db_manager.add_message(base_session, "user", request.message)

    async def event_generator():
        try:
            config = {"configurable": {"thread_id": execution_thread_id}}
            final_ans = ""
            
            if request.approve_web_search is not None:
                # Update checkpoint state with human confirmation data and cleanly resume execution
                agent.graph.update_state(config, {"user_approved_web_search": request.approve_web_search}, as_node="ask_user")
                async for event in agent.graph.astream(None, config, stream_mode="values"):
                    if isinstance(event, dict) and "final_answer" in event:
                        final_ans = _sanitize_ai_response(event.get("final_answer", ""))
            else:
                async for event in agent.graph.astream({"question": request.message, "session_id": base_session}, config, stream_mode="values"):
                    if isinstance(event, dict) and "final_answer" in event:
                        final_ans = _sanitize_ai_response(event.get("final_answer", ""))

            # Failure and fallthrough checking
            failure_indicators = ["unable to extract", "i don't know", "cannot find", "not mentioned"]
            is_failure = any(p in final_ans.lower() for p in failure_indicators)

            # Check checkpoint tracking state to see if graph hit an active interrupt boundary
            state = agent.graph.get_state(config)
            is_interrupted = len(state.next) > 0 and state.next[0] == "ask_user"

            if is_interrupted or ((not final_ans or len(final_ans) < 5 or is_failure) and request.approve_web_search is None):
                yield _to_sse("approval_required", {
                    "message": "I could not find an answer in the documents. Would you like to check the web?",
                    "session_id": base_session
                })
                return

            if final_ans:
                db_manager.add_message(base_session, "assistant", final_ans)
            
            yield _to_sse("done", {"message": final_ans, "session_id": base_session})
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield _to_sse("error", {"detail": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health(): 
    return {"status": "ok", "database": "connected"}

@app.get("/conversations")
async def get_conversations(): 
    return db_manager.get_detailed_sessions()

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str): 
    return [{"role": r.role, "content": r.content} for r in db_manager.get_messages(session_id)]

@app.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    parsed_doc = doc_processor.parse(file.filename, file.content_type, content)
    db_manager.add_document("global_kb", parsed_doc.content)
    doc_processor.index_to_pinecone("global_kb", parsed_doc)
    return {"message": "Indexed", "session_id": session_id}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    db_manager.delete_session(session_id)
    return {"message": "Deleted"}