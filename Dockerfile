FROM python:3.10-slim

# Install C-libraries for fast audio IO and WORLD vocoder compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# 1 worker with 2 threads prevents RAM multiplication on Render's 512MB tier
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 300 app:app"
