import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use local cache directory instead of /mnt (which is read-only on macOS)
PROJECT_ROOT = Path(BASE_DIR).parent
CACHE_DIR = PROJECT_ROOT / ".cache"
MOUNT_DIR = Path("/mnt")  # Keep for reference, but use CACHE_DIR for local dev

# Load .env file variables (if present)
config = dotenv_values(".env")

# Default to local cache directory, but allow override via environment variables
config.update({
    "HF_HOME": os.getenv("HF_HOME", str(CACHE_DIR / "hf_cache")),
    "TORCH_HOME": os.getenv("TORCH_HOME", str(CACHE_DIR / "torch_cache")),
    "VECTOR_STORE_PATH": os.getenv("VECTOR_STORE_PATH", str(CACHE_DIR / "vector_store")),
    "DOCUMENTS_PATH": os.getenv("DOCUMENTS_PATH", str(CACHE_DIR / "documents")),
    "MODEL_PATH": os.getenv("MODEL_PATH", str(CACHE_DIR / "models")),
    "GGUF_MODELS_PATH": os.getenv("GGUF_MODELS_PATH", str(CACHE_DIR / "models/gguf")),
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512"),
    # NEW: database URL for user-recognition + memory
    # For dev, SQLite in the local .cache dir; override in .env for Postgres, etc.
    "DATABASE_URL": os.getenv(
        "DATABASE_URL",
        f"sqlite:///{CACHE_DIR / 'furhat_memory.db'}"
    ),
})

# Set environment variables from config
for key, value in config.items():
    os.environ[key] = str(value)
