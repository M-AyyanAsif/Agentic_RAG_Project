"""
Indus-Guardian Frontend - Optimized
- High-Performance SSE Streaming
- Human-in-the-loop Approval System
- Fixed Connection Logic
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
# Hardcoding to 127.0.0.1 for maximum stability on local Windows dev
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Indus-Guardian AI",
    page_icon="🛡️",
    layout="wide",
)

# --- STYLING ---
def load_ui_theme() -> None:
    st.markdown("""
        <style>
            [data-testid="stChatMessage"] {
                color: white !important;
                border-radius: 10px;
            }
            .stChatMessage.assistant {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid #30363d;
            }
            h1 { color: #58a6ff !important; }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

load_ui_theme()

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.chat_messages = []
    st.session_state.approval_needed = False
    st.session_state.pending_prompt = ""
    st.session_state.thought_logs = []

# --- API CORE FUNCTIONS ---

def stream_answer(message: str, approve_web_search: bool | None = None):
    """Handles the SSE stream from FastAPI."""
    payload = {
        "session_id": st.session_state.session_id, 
        "message": message, 
        "approve_web_search": approve_web_search
    }
    
    url = f"{API_BASE}/chat/stream"
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as response:
            if not response.ok:
                yield f"🚨 Backend Error: {response.status_code} - {response.text}"
                return
            
            event_type = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                
                if raw_line.startswith("event: "): 
                    event_type = raw_line.replace("event: ", "").strip()
                elif raw_line.startswith("data: "):
                    data_str = raw_line.replace("data: ", "").strip()
                    try:
                        data = json.loads(data_str)
                        
                        if event_type == "thought": 
                            st.session_state.thought_logs.append(data.get("log", ""))
                        elif event_type == "token":
                            yield data.get("token", "")
                        elif event_type == "approval_required":
                            st.session_state.approval_needed = True
                            st.session_state.pending_prompt = message
                            # Stop the stream here to show UI buttons
                            return
                        elif event_type == "done" and data.get("message"):
                            yield data["message"]
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield "❌ Connection failed: Is the Backend running on port 8000?"
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def upload_document(uploaded_file):
    """Handles file uploads to the backend."""
    url = f"{API_BASE}/upload"
    try:
        # Prepare the file and data payload
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"session_id": st.session_state.session_id}
        
        response = requests.post(url, files=files, data=data, timeout=30)
        return response
    except Exception as e:
        st.error(f"Upload logic failed: {e}")
        return None

# --- SIDEBAR UI ---
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
        with st.spinner("Processing..."):
            res = upload_document(uploaded)
            if res and res.status_code == 200:
                st.success(f"✅ {uploaded.name} indexed!")
            elif res:
                st.error(f"Error {res.status_code}: {res.text}")
            else:
                st.error("Backend unreachable. Check your terminal.")

    st.divider()
    st.subheader("⚙️ Agent Reasoning")
    # Show the last 5 thoughts to see what the agent is doing
    for log in st.session_state.thought_logs[-5:]:
        st.caption(f"🧠 {log}")

# --- MAIN CHAT UI ---
st.markdown("<h1 style='text-align: center;'>🛡️ Indus-Guardian AI</h1>", unsafe_allow_html=True)
st.caption(f"📍 Session ID: {st.session_state.session_id} | 🕒 {dt.datetime.now().strftime('%H:%M')}")

# Render history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Approval UI Logic
if st.session_state.approval_needed:
    st.warning("💡 Not found in documents. Search web?")
    col1, col2 = st.columns(2)
    if col1.button("✅ Yes, Search Web", use_container_width=True):
        st.session_state.approval_needed = False
        with st.chat_message("assistant"):
            response = st.write_stream(stream_answer(st.session_state.pending_prompt, approve_web_search=True))
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()
    if col2.button("❌ No, Stay Local", use_container_width=True):
        st.session_state.approval_needed = False
        st.session_state.chat_messages.append({"role": "assistant", "content": "Restricted to local docs."})
        st.rerun()

# Chat Input Logic
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt))
    
    if not st.session_state.approval_needed:
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    else:
        st.rerun()