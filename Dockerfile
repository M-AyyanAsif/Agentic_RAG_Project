# Sync with backend for consistency and Python 3.13 speed
FROM python:3.13-slim

# Combine Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies (combined from both files)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy both requirements files first to leverage Docker cache
COPY backend/requirements.txt /app/backend/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt

# Install all dependencies with high timeout for network stability
RUN pip install --upgrade pip && \
    pip install --default-timeout=2000 --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --default-timeout=2000 --no-cache-dir -r /app/frontend/requirements.txt

# Copy the entire source code for both services
COPY backend /app/backend
COPY frontend /app/frontend

# Expose Hugging Face default port (7860) and internal API port (8000)
EXPOSE 7860 8000

# Start both services: FastAPI runs in the background (&), Streamlit runs in the foreground
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"]