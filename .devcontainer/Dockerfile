# -------------------------------------------------------------
# Stage 1: Base Python environment with build tools
# -------------------------------------------------------------
    FROM python:3.11-slim

    # System dependencies for building and for Ollama
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget cmake clang libc++-dev libc++abi-dev \
        libomp-dev ninja-build python3-dev libopenblas-dev pkg-config ca-certificates \
      && rm -rf /var/lib/apt/lists/*
    
    # -------------------------------------------------------------
    # Install Poetry and project dependencies
    # -------------------------------------------------------------
    RUN pip install --upgrade pip setuptools wheel && pip install poetry==2.0.1
    WORKDIR /app
    
    # Copy Poetry files
    COPY pyproject.toml poetry.lock* /app/
    RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi
    
    # Optional extra dependencies
    COPY requirements_poetry.txt /app/
    RUN if [ -f requirements_poetry.txt ]; then pip install --no-cache-dir -r requirements_poetry.txt; fi
    
    # -------------------------------------------------------------
    # Install Ollama
    # -------------------------------------------------------------
    # Ollama official Linux install script:
    RUN curl -fsSL https://ollama.com/install.sh | sh
    
    # Add Ollama binary to PATH (just to be sure)
    ENV PATH="/usr/local/bin:${PATH}"
    
    # -------------------------------------------------------------
    # (Optional) Install llama-cpp-python if you still want local GGUF support
    # Comment out if you’ll use Ollama exclusively
    # RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python==0.3.7
    
    # -------------------------------------------------------------
    # Copy the rest of your app
    # -------------------------------------------------------------
    COPY . /app
    
    # Environment variables
    ENV OLLAMA_BASE_URL="http://localhost:11434"
    ENV HF_HOME=/app/.cache/hf
    ENV TRANSFORMERS_CACHE=/app/.cache/hf
    ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/st
    
    # Expose the FastAPI port
    EXPOSE 8000
    
    # -------------------------------------------------------------
    # Entrypoint: start Ollama service + your API
    # -------------------------------------------------------------
    # Ollama runs as a background service; we then start FastAPI
    CMD bash -c "\
        echo 'Starting Ollama service...' && \
        ollama serve & \
        sleep 3 && \
        echo 'Starting FastAPI app...' && \
        uvicorn main:app --host 0.0.0.0 --port 8000 \
    "
