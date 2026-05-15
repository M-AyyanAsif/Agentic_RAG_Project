"""
Indus-Guardian Frontend
- Optimized for Python 3.13
- High-Performance SSE Streaming
- Human-in-the-loop Approval System
"""

from __future__ import annotations
import datetime as dt
import json
import os
from typing import Any
from uuid import uuid4
import requests
import streamlit as st

# --- CONFIGURATION ---
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Indus-Guardian AI",
    page_icon="🛡️",
    layout="wide",
)

# --- STYLING ---
def load_ui_theme() -> None:
    st.markdown("""
        <style>
            /* Assistant Message Contrast */
            [data-testid="stChatMessage"] {
                color: white !important;
                border-radius: 10px;
            }
            .stChatMessage.assistant {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid #30363d;
            }
            /* Clean Headers */
            h1 { color: #58a6ff !important; }
            /* Hide Streamlit Branding for Professional Look */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

load_ui_theme()

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.chat_messages = []
    st.session_state.approval_needed = False
    st.session_state.pending_prompt = ""
    st.session_state.thought_logs = []

# --- API CORE ---
def stream_answer(message: str, approve_web_search: bool | None = None):
    """Handles the SSE stream from FastAPI."""
    payload = {
        "session_id": st.session_state.session_id, 
        "message": message, 
        "approve_web_search": approve_web_search
    }
    
    try:
        # 180s timeout allows the LLM to think deeply without the connection dropping
        with requests.post(f"{API_BASE}/chat/stream", json=payload, stream=True, timeout=180) as response:
            if not response.ok:
                yield f"🚨 Backend Error: {response.status_code}"
                return
            
            event_type = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line: continue
                
                if raw_line.startswith("event: "): 
                    event_type = raw_line.replace("event: ", "")
                elif raw_line.startswith("data: "):
                    data_str = raw_line.replace("data: ", "")
                    data = json.loads(data_str)
                    
                    if event_type == "thought": 
                        st.session_state.thought_logs.append(data.get("log", ""))
                    elif event_type == "token":
                        yield data.get("token", "")
                    elif event_type == "approval_required":
                        st.session_state.approval_needed = True
                        st.session_state.pending_prompt = message
                    elif event_type == "done" and data.get("message"):
                        yield data["message"]
    except Exception as e:
        yield f"❌ Connection failed: Please check if the Docker containers are running. Error: {str(e)}"

# --- SIDEBAR & CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("Admin Controls")
    
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.chat_messages = []
        st.session_state.thought_logs = []
        st.session_state.approval_needed = False
        st.rerun()

    st.divider()
    st.subheader("📄 Knowledge Base")
    uploaded = st.file_uploader("Upload document for RAG", type=["pdf", "docx"])
    if uploaded and st.button("📥 Index into System", use_container_width=True):
        with st.spinner("Processing PDF..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                res = requests.post(f"{API_BASE}/upload", files=files, data={"session_id": st.session_state.session_id})
                if res.ok: st.success("Document analyzed successfully!")
                else: st.error("Failed to index document.")
            except: st.error("Backend unreachable.")

    st.divider()
    st.subheader("⚙️ Agent Reasoning")
    for log in st.session_state.thought_logs[-3:]:
        st.caption(f"🧠 {log}")

# --- MAIN CHAT UI ---
st.markdown("<h1 style='text-align: center;'>🛡️ Indus-Guardian AI</h1>", unsafe_allow_html=True)
st.caption(f"📍 Session ID: {st.session_state.session_id} | 🕒 {dt.datetime.now().strftime('%H:%M')}")

# Render message history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Approval UI (Human-in-the-loop)
if st.session_state.approval_needed:
    st.info("💡 I couldn't find the answer in your documents. Would you like me to search the web?")
    a1, a2 = st.columns(2)
    if a1.button("✅ Yes, search the internet", use_container_width=True):
        st.session_state.approval_needed = False
        with st.chat_message("assistant"):
            response = st.write_stream(stream_answer(st.session_state.pending_prompt, approve_web_search=True))
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()
    if a2.button("❌ No, stay local only", use_container_width=True):
        st.session_state.approval_needed = False
        st.session_state.chat_messages.append({"role": "assistant", "content": "Understood. I will restrict my knowledge to your uploaded documents."})
        st.rerun()

# Chat Input Logic
if prompt := st.chat_input("Query your documents..."):
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate Assistant response via Stream
    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt))
    
    # Store history only if we aren't waiting for a web search approval
    if not st.session_state.approval_needed:
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    else:
        st.rerun() # Force UI update to show approval buttons