# ====================================================================
# AI-Powered Connected Vehicle Analytics Platform (BDA)
# Production Container Specification
# ====================================================================
FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install minimal OS dependencies for health checking and C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application directories, pre-trained ML models, configurations, and verification suites
COPY config/ config/
COPY data/ data/
COPY data_generator/ data_generator/
COPY ml_pipeline/ ml_pipeline/
COPY models/ models/
COPY reports/ reports/
COPY dashboard/ dashboard/
COPY notebooks/ notebooks/
COPY verify_vehicle_configurations.py .
COPY verify_all_phases.py .
COPY verify_phase_a.py .
COPY mlruns/ mlruns/
COPY mlflow.db .

# Ensure required runtime directories exist inside container
RUN mkdir -p data/landing reports

# Expose Streamlit web application port
EXPOSE 8501

# Health check using Streamlit's internal health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit Fleet Monitoring Dashboard entry point
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
