FROM python:3.11-slim

# ============================================================
# PYTHON SETTINGS
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Reduce CPU thread usage and memory overhead.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Hugging Face / tokenizer settings
ENV TOKENIZERS_PARALLELISM=false

# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION
# ============================================================

# ============================================================
# APPLICATION
# ============================================================

# ============================================================
# APPLICATION
# ============================================================

COPY . .

# Verify frontend is included in the Docker image
RUN echo "===== FRONTEND CHECK =====" \
    && pwd \
    && ls -la /app \
    && ls -la /app/frontend \
    && test -f /app/frontend/index.html \
    && echo "===== frontend/index.html FOUND ====="
# ============================================================
# RENDER PORT
# ============================================================

EXPOSE 10000

# ============================================================
# START APPLICATION
# ============================================================

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]