"""Streamlit frontend for Innocent AI."""

from __future__ import annotations
import datetime as dt
import json
import os
from typing import Any
from uuid import uuid4
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "").rstrip("/")
API_CANDIDATES = [
    API_BASE,
    "http://backend:8000",  # Docker service-to-service
    "http://localhost:8000",  # Local run
    "http://127.0.0.1:8000",
]

st.set_page_config(
    page_title="Innocent AI",
    page_icon="🤖",
    layout="wide",
)

def load_css() -> None:
    # SENIOR FIX: Added CSS to force Assistant (AI) messages to be white
    ai_style = """
    <style>
        /* Force AI message text to white */
        [data-testid="stChatMessage"] {
            color: white !important;
        }
        /* Specifically target the assistant messages if needed */
        .stChatMessage.assistant {
            background-color: rgba(255, 255, 255, 0.05);
        }
        /* Ensure markdown inside messages is white */
        [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }
    </style>
    """
    st.markdown(ai_style, unsafe_allow_html=True)
    
    try:
        if os.path.exists("styles.css"):
            with open("styles.css", "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

load_css()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.chat_messages = []
    st.session_state.approval_needed = False
    st.session_state.pending_prompt = ""
    st.session_state.last_prompt = ""
    st.session_state.thought_logs = []
    st.session_state.available_sessions = {}
    st.session_state.api_base = ""
    st.session_state.history_menu_for = ""

if "history_menu_for" not in st.session_state:
    st.session_state.history_menu_for = ""


def get_working_api_base() -> str:
    if st.session_state.api_base:
        return st.session_state.api_base

    for candidate in API_CANDIDATES:
        if not candidate:
            continue
        try:
            resp = requests.get(f"{candidate}/health", timeout=2)
            if resp.ok:
                st.session_state.api_base = candidate
                return candidate
        except requests.RequestException:
            continue
    return ""


def safe_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    base = get_working_api_base()
    if not base:
        raise requests.RequestException(
            "Backend unreachable. Start API server or set API_BASE correctly."
        )
    return requests.request(method=method, url=f"{base}{path}", **kwargs)

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🤖 Innocent AI</h1>", unsafe_allow_html=True)
st.caption(f"🕒 Current Time: {dt.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

# Helper: Fetch Sessions with Previews
def refresh_session_list():
    try:
        resp = safe_request("GET", "/sessions", timeout=5)
        if resp.ok:
            session_ids = resp.json().get("sessions", [])
            session_previews = {}
            for s_id in session_ids:
                msg_resp = safe_request("GET", f"/sessions/{s_id}/messages", timeout=2)
                if msg_resp.ok:
                    msgs = msg_resp.json().get("messages", [])
                    user_messages = [m for m in msgs if m.get("role") == "user"]
                    if user_messages:
                        title = " ".join(user_messages[0]["content"].split()[:6]).strip()
                    else:
                        title = "Untitled chat"
                    session_previews[s_id] = {
                        "session_id": s_id,
                        "title": title or "Untitled chat",
                        "message_count": len(msgs),
                    }
            st.session_state.available_sessions = session_previews
    except requests.RequestException:
        pass

if not st.session_state.available_sessions:
    refresh_session_list()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("💬 Session Controls")

    if st.button("➕ Add Chat", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.chat_messages = []
        st.session_state.thought_logs = []
        st.session_state.approval_needed = False
        st.session_state.pending_prompt = ""
        st.rerun()

    if st.button("🗑 Delete Current Chat", use_container_width=True):
        try:
            safe_request(
                "DELETE",
                "/sessions",
                json={"session_id": st.session_state.session_id},
                timeout=5,
            )
            st.session_state.chat_messages = []
            refresh_session_list()
            st.rerun()
        except requests.RequestException:
            st.error("Delete failed.")

    if st.button("🔄 Refresh Sessions", use_container_width=True):
        refresh_session_list()
        st.rerun()

    st.subheader("🕘 Chat History")
    if st.session_state.available_sessions:
        for s_id, info in st.session_state.available_sessions.items():
            c1, c2 = st.columns([5, 1])
            with c1:
                label = f"{info['title']} ({s_id[:8]})"
                if st.button(label, key=f"load_{s_id}", use_container_width=True):
                    st.session_state.session_id = s_id
                    history_resp = safe_request(
                        "GET", f"/sessions/{s_id}/messages", timeout=5
                    )
                    if history_resp.ok:
                        st.session_state.chat_messages = history_resp.json().get(
                            "messages", []
                        )
                        st.session_state.approval_needed = False
                        st.session_state.pending_prompt = ""
                        st.rerun()
            with c2:
                if st.button("⋯", key=f"menu_{s_id}", use_container_width=True):
                    st.session_state.history_menu_for = (
                        "" if st.session_state.history_menu_for == s_id else s_id
                    )
                    st.rerun()
            if st.session_state.history_menu_for == s_id:
                if st.button(
                    "Delete permanently",
                    key=f"delete_hist_{s_id}",
                    use_container_width=True,
                ):
                    safe_request(
                        "DELETE",
                        "/sessions",
                        json={"session_id": s_id},
                        timeout=5,
                    )
                    if st.session_state.session_id == s_id:
                        st.session_state.session_id = str(uuid4())
                        st.session_state.chat_messages = []
                    st.session_state.history_menu_for = ""
                    refresh_session_list()
                    st.rerun()
    else:
        st.caption("No previous chats yet.")

    st.markdown("---")
    st.subheader("📄 Upload Knowledge")
    uploaded = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])

    if uploaded and st.button("📥 Index Document", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = safe_request(
                    "POST",
                    "/upload",
                    files=files,
                    data={"session_id": st.session_state.session_id},
                    timeout=120,
                )
                if resp.ok: 
                    st.success("Document Indexed!")
                else:
                    st.error(f"Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    st.markdown("---")
    st.subheader("🧠 Current Session")
    st.caption(st.session_state.session_id)

    st.subheader("⚙ Thought Process")
    for line in st.session_state.thought_logs[-5:]:
        st.caption(f"• {line}")

# -----------------------------
# Chat Interface Logic
# -----------------------------
def stream_answer(message: str, approve_web_search: bool | None = None):
    payload = {"session_id": st.session_state.session_id, "message": message, "approve_web_search": approve_web_search}
    try:
        with safe_request(
            "POST", "/chat/stream", json=payload, stream=True, timeout=180
        ) as response:
            if not response.ok:
                yield f"Error from backend: {response.status_code}"
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
                        st.session_state.thought_logs.append(data["log"])
                    elif event_type == "token":
                        yield data["token"]
                    elif event_type == "approval_required":
                        st.session_state.approval_needed = True
                        st.session_state.pending_prompt = message
                    elif event_type == "done" and data.get("message"):
                        yield data["message"]
    except Exception:
        yield "Backend connection failed: please ensure backend is running."

# Render Messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.approval_needed:
    st.info("I could not find this in uploaded docs. Search internet?")
    a1, a2 = st.columns(2)
    if a1.button("Yes, search web", use_container_width=True):
        with st.chat_message("assistant"):
            response = st.write_stream(
                stream_answer(st.session_state.pending_prompt, approve_web_search=True)
            )
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.approval_needed = False
        st.session_state.pending_prompt = ""
        st.rerun()
    if a2.button("No", use_container_width=True):
        st.session_state.chat_messages.append({"role": "assistant", "content": "OK"})
        st.session_state.approval_needed = False
        st.session_state.pending_prompt = ""
        st.rerun()

# Chat Input
if prompt := st.chat_input("Ask questions about your uploaded documents..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt))
    st.session_state.chat_messages.append({"role": "assistant", "content": response})