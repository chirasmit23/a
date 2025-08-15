# --------------------------
# 1. Base image
# --------------------------
FROM python:3.10

# --------------------------
# 2. Install system dependencies
# --------------------------
# - tesseract-ocr for OCR
# - ffmpeg for yt-dlp
# - curl/wget/git for general utility
# --------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    yt-dlp \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# 3. Set work directory
# --------------------------
WORKDIR /app

# --------------------------
# 4. Copy requirements & install Python deps
# --------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# 5. Copy project files
# --------------------------
COPY . .

# --------------------------
# 6. Environment variables
# --------------------------
# Render sets PORT automatically
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# --------------------------
# 7. Expose port
# --------------------------
EXPOSE 10000

# --------------------------
# 8. Run Streamlit app
# --------------------------
CMD ["streamlit", "run", "app.py", "--server.port", "10000", "--server.address", "0.0.0.0"]
