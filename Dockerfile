# ===== RUNTIME BASE =====
FROM python:3.11-slim

# OS deps ringan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Buat user non-root
RUN useradd -m app
WORKDIR /app

# ===== PYTHON DEPS =====
# (biar cache build efektif)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== APP CODE & ASSETS =====
# copy backend, frontend, dan model
COPY backend ./backend
COPY frontend ./frontend
COPY fake_news_detector_distilbert ./fake_news_detector_distilbert

# Env dasar (opsional)
ENV PYTHONUNBUFFERED=1
# Port di container
EXPOSE 4000

# Pindah ke user non-root
USER app

# Jalankan uvicorn
# NOTE: Pastikan import path ini benar (backend.main:app)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "4000"]
