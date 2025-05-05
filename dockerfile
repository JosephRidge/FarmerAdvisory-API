FROM ubuntu:22.04

# System setup with essential dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    cmake \
    build-essential \
    libgl1 \          # Needed for some CV/ML packages
    git \             # Required for some pip installations
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with cache optimization
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu/ \
    -r requirements.txt

# Copy application files (exclude unnecessary files with .dockerignore)
COPY . .

# Runtime optimizations
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    GGML_CUBLAS=0 \
    PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE="/app/model-cache" \
    XET_CACHE_DIR="/app/xet-cache"

# Create cache directories (avoids permission issues)
RUN mkdir -p /app/model-cache /app/xet-cache && \
    chmod -R 777 /app/model-cache /app/xet-cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]