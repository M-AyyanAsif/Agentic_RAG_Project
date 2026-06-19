"""
Indus-Guardian Frontend - Clean UI Engine
- High-Performance SSE Streaming
- Decoupled Ingestion & Global Knowledge Base Routing
- Production-Grade Recruiter Demo Layout
"""

from __future__ import annotations
import datetime as dt
import json
import os
import re
from typing import Any
from uuid import uuid4
import requests
import streamlit as st

# --- CONFIGURATION ---
# When deploying to Hugging Face, change this to your production API URL if separated, 
# or keep it as local/container relative mapping.
API_BASE = "http://127.0.0.1:8000"
TARGET_NAMESPACE = "global_knowledge_base"

st.set_page_config(
    page_title="Indus-Guardian AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED UI STYLING ---
def load_ui_theme() -> None:
    st.markdown("""
        <style>
            [data-testid="stChatMessage"] {
                color: white !important;
                border-radius: 10px;
            }
            .stChatMessage.assistant {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid #30363d;
            }
            h1 { color: #58a6ff !important; font-weight: 700; }
            h3 { color: #8b949e !important; }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            /* Custom Status Cards */
            .status-card {
                background-color: #0d1117;
                border: 1px solid #21262d;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 5px;
            }

            /* --- SIDEBAR COMPACT OPTIMIZATION --- */
            /* Completely eliminate blank vertical padding from the top of the sidebar layout */
            [data-testid="stSidebarUserContent"] {
                padding-top: 0rem !important;
                padding-bottom: 1rem !important;
            }
            
            /* Override internal container spacing to push items upward */
            [data-testid="stSidebar"] > div {
                padding-top: 0rem !important;
            }
            
            /* Minimize spacing between consecutive sidebar widgets and items */
            [data-testid="stSidebar"] .stElementContainer {
                margin-bottom: 0.3rem !important;
            }
            
            /* Compress divider margins inside the sidebar wrapper */
            [data-testid="stSidebar"] hr {
                margin-top: 0.4rem !important;
                margin-bottom: 0.4rem !important;
            }
            
            /* Remove top margin from the title header to stick it to the top */
            [data-testid="stSidebar"] h3 {
                margin-top: 0rem !important;
                margin-bottom: 0.3rem !important;
                padding-top: 0rem !important;
            }
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

def get_history_sessions() -> list[dict[str, str]]:
    """Fetches session titles mapping directly from backend sqlite."""
    try:
        # CHANGED: Increased timeout slightly for reliability
        res = requests.get(f"{API_BASE}/conversations", timeout=15)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def get_session_messages(session_id: str) -> list[dict[str, str]]:
    """Fetches the actual text chat history for a selected session."""
    try:
        # CHANGED: Increased timeout slightly for reliability
        res = requests.get(f"{API_BASE}/sessions/{session_id}/messages", timeout=15)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def delete_history_session(session_id: str) -> bool:
    """Deletes complete session row inside db endpoints."""
    try:
        # CHANGED: Increased timeout slightly for reliability
        res = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=15)
        return res.status_code == 200
    except Exception:
        pass
    return False

def stream_answer(message: str, approve_web_search: bool | None = None):
    """Handles the SSE stream from FastAPI and strips raw structural data formats."""
    payload = {
        "session_id": TARGET_NAMESPACE, 
        "conversation_id": st.session_state.session_id, 
        "message": message, 
        "approve_web_search": approve_web_search
    }
    
    url = f"{API_BASE}/chat/stream"
    
    try:
        # CHANGED: Bumped timeout from 180 to 300 seconds to give deep-doc queries time to stream safely
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
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
                            return
                        elif event_type == "done" and data.get("message"):
                            final_msg = data["message"]
                            if isinstance(final_msg, str) and (final_msg.startswith("[{'type':") or final_msg.startswith("{'type':")):
                                matches = re.findall(r"'text':\s*'(.*?)'(?:,\s*'extras'|\s*})", final_msg, re.DOTALL)
                                if matches:
                                    final_msg = "\n".join(matches)
                            yield ("__FINAL_DONE_FLAG__" + final_msg)
                    except json.JSONDecodeError:
                        continue
    # CHANGED: Added specific ReadTimeout handler to differentiate from standard Connection issues
    except requests.exceptions.ReadTimeout:
        yield "❌ Timeout: The backend took too long to analyze the documents. Please try again."
    except requests.exceptions.ConnectionError:
        yield "❌ Connection failed: Is the Backend engine running on port 8000?"
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def upload_document(uploaded_file):
    """Handles file uploads to the global Knowledge Base."""
    url = f"{API_BASE}/upload"
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"session_id": TARGET_NAMESPACE} 
        
        # CHANGED: Extended upload timeout to 300 to match stream limits
        response = requests.post(url, files=files, data=data, timeout=300)
        return response
    except Exception as e:
        st.error(f"Upload logic failed: {e}")
        return None

# ==========================================
# SECTION B: SIDEBAR (SECONDARY CONTROLS)
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 Indus Guardian")
    
    # Cloud-Native Architecture Status Card
    st.markdown(f"""
    <div class="status-card">
        <p style="margin:0; font-weight:bold; color:#58a6ff;">Cloud-Native Hybrid RAG</p>
        <p style="margin:5px 0 0 0; font-size:12px; color:#7ee787;">● Vector Database Connected</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Core pre-indexed notice
    st.success("✅ AI/ML Reference Documents Pre-Indexed")

    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.chat_messages = []
        st.session_state.thought_logs = []
        st.session_state.approval_needed = False
        st.rerun()

    st.divider()
    
    # Interactive Testing Uploader
    st.markdown("### 📄 Test Unseen Data")
        
    uploaded = st.file_uploader("Upload new document for live RAG", type=["pdf", "docx"], label_visibility="collapsed")
    
    if uploaded:
        if st.button("📥 Parse & Embed Document", use_container_width=True):
            with st.spinner("Processing embeddings synchronously..."):
                res = upload_document(uploaded)
                if res and res.status_code == 200:
                    st.success(f"Successfully integrated '{uploaded.name}' into the production index!")
                elif res:
                    st.error(f"Ingestion Aborted ({res.status_code}): {res.text}")
                else:
                    st.error("Server connection timeout.")

    st.divider()
    
    # Conversation History Section
    st.markdown("### 💬 Session History")
    sessions = get_history_sessions()
    
    if not sessions:
        st.caption("No active histories saved in SQLite backend.")
    else:
        for index, s in enumerate(sessions):
            target_id = s["session_id"]
            raw_title = s["title"].strip()
            display_title = raw_title[:22] + "..." if len(raw_title) > 22 else raw_title
            if not display_title:
                display_title = f"Session_{target_id[:6]}"
                
            col_text, col_menu = st.columns([0.82, 0.18])
            
            with col_text:
                if st.button(f"💬 {display_title}", key=f"sel_btn_{target_id}_{index}", use_container_width=True):
                    st.session_state.session_id = target_id
                    st.session_state.chat_messages = get_session_messages(target_id)
                    st.session_state.thought_logs = ["Context restored successfully."]
                    st.session_state.approval_needed = False
                    st.rerun()
            
            with col_menu:
                with st.popover("⋮", key=f"pop_{target_id}_{index}"):
                    if st.button("🗑️ Delete Session", key=f"del_act_{target_id}_{index}", use_container_width=True):
                        if delete_history_session(target_id):
                            if st.session_state.session_id == target_id:
                                st.session_state.session_id = str(uuid4())
                                st.session_state.chat_messages = []
                                st.session_state.thought_logs = []
                            st.toast("Session cleared from cache.")
                            st.rerun()

    st.divider()
    st.markdown("### 🧠 Agent Execution Log")
    for log in st.session_state.thought_logs[-3:]:
        st.caption(f"⚙️ {log}")


# ==========================================
# SECTION A: MAIN CHAT INTERFACE (PRIMARY FOCUS)
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🤖 Indus-Guardian AI</h1>", unsafe_allow_html=True)

# Render conversation history blocks
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        content_to_display = msg["content"]
        
        if isinstance(content_to_display, str) and ("{'type':" in content_to_display or "'extras':" in content_to_display):
            matches = re.findall(r"'text':\s*\"(.*?)\"(?:,\s*'extras'|\s*})", content_to_display, re.DOTALL)
            if not matches:
                matches = re.findall(r"'text':\s*'(.*?)'(?:,\s*'extras'|\s*})", content_to_display, re.DOTALL)
            if matches:
                content_to_display = "\n".join(matches).replace("\\n", "\n")

        content_to_display = content_to_display.strip()
        if (content_to_display.startswith("'") and content_to_display.endswith("'")) or (content_to_display.startswith('"') and content_to_display.endswith('"')):
            content_to_display = content_to_display[1:-1]
            
        content_to_display = content_to_display.replace("\\n", "\n")
        st.markdown(content_to_display)

# Web Search Intent Fallback UI
if st.session_state.approval_needed:
    st.warning("💡 Query context outside of Reference Guides. Activate Web Search Engine?")
    col1, col2 = st.columns(2)
    if col1.button("✅ Yes, Trigger Web Agents", use_container_width=True):
        st.session_state.approval_needed = False
        with st.chat_message("assistant"):
            placeholder = st.empty()
            accumulated_response = ""
            for chunk in stream_answer(st.session_state.pending_prompt, approve_web_search=True):
                if chunk.startswith("__FINAL_DONE_FLAG__"):
                    accumulated_response = chunk.replace("__FINAL_DONE_FLAG__", "")
                    break
                accumulated_response += chunk
                placeholder.markdown(accumulated_response + "▌")
            placeholder.markdown(accumulated_response)
            st.session_state.chat_messages.append({"role": "assistant", "content": accumulated_response})
        st.rerun()
        
    if col2.button("❌ No, Strict to KB Mode", use_container_width=True):
        st.session_state.approval_needed = False
        st.session_state.chat_messages.append({"role": "assistant", "content": "Query evaluation restricted. Search strictly limited to local cloud indices."})
        st.rerun()

# Primary Question Capture Box
if prompt := st.chat_input("Ask Indus Guardian "):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated_response = ""
        
        for chunk in stream_answer(prompt):
            if chunk.startswith("__FINAL_DONE_FLAG__"):
                accumulated_response = chunk.replace("__FINAL_DONE_FLAG__", "")
                break
            accumulated_response += chunk
            placeholder.markdown(accumulated_response + "▌")
            
        placeholder.markdown(accumulated_response)
    
    if not st.session_state.approval_needed:
        st.session_state.chat_messages.append({"role": "assistant", "content": accumulated_response})
    else:
        st.rerun()