# docker/Dockerfile
# Base image — Python 3.11 slim (small, production-grade)
FROM python:3.11-slim

# Install system dependencies
# Tesseract for OCR, libglib for OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caching — if requirements don't change,
# this layer is cached and pip install doesn't re-run on every build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/

# Expose port
EXPOSE 8000

# Health check — Docker will ping this every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
