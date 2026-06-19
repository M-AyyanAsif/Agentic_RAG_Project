# Indus-Guardian: Agentic Intelligence Project

Agentic RAG platform designed for privacy-conscious deployments.  
This project demonstrates production architecture for internship  using FastAPI, LangGraph, hybrid retrieval, and a modern Streamlit chat UI.

## Why This Project
- Uses a transparent LangGraph state machine to explain every agent step.
- Supports PDF and DOCX document ingestion for practical enterprise workflows.
- Implements HITL consent before web search to preserve data sovereignty.

## Architecture Overview
- `backend/main.py`: FastAPI async API with SSE token streaming.
- `backend/engine/graph.py`: LangGraph nodes: Ingest -> Cache -> Retrieve -> Grade -> User Approval -> Web Search -> Refine -> Generate.
- `backend/engine/retrieval_service.py`: Hybrid retrieval + CrossEncoder rerank.
- `backend/core/database.py`: SQLite chat history + semantic cache.
- `frontend/app.py`: Streamlit dark UI with `st.chat_message` and `st.write_stream`.
- `deployment/`: Dockerfiles and Compose for reproducible multi-service startup.

## Quick Start (Docker)
1. Copy `.env.example` to `.env` and fill your keys.
2. Run:
   ```bash
   cd deployment
   docker compose up --build
   ```
3. Open:
   - Frontend: [http://localhost:8501](http://localhost:8501)
   - Backend docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Windows One-Command Startup
- Use the provided script from project root:
  - `.\run.ps1 -Build` (first run)
  - `.\run.ps1` (next runs)
- A copy-ready machine-specific template is included at `.env.template.windows`.

## Reliability and Operations
- Secrets are excluded through `.gitignore` (`.env` is never committed).
- Dependencies are pinned in both `backend/requirements.txt` and `frontend/requirements.txt`.
- Rotating logs are enabled at `backend/logs/app.log` (5 files, 5MB each).
- Containers are configured with `restart: unless-stopped`.
- SQLite backups:
  - Manual: `python scripts/backup_db.py`
  - Recommended: run via Windows Task Scheduler daily.

## Testing and CI
- Test suite includes:
  - Document parsing validation
  - Graph routing behavior (relevant vs non-relevant)
  - API smoke tests
- Run locally:
  - `pytest backend/tests`
- CI workflow is in `.github/workflows/ci.yml`:
  - Ruff lint
  - Pytest
  - Docker compose config validation

## Local-First Privacy Positioning 
- Document uploads stay within your controlled infra.
- External web search is user-approved (HITL consent loop).
- Session data is stored in SQLite for clear, auditable ownership.
- Can be deployed on local laptops, private servers, or on-prem clusters.

## Made by love from Muhammad Ayyan
