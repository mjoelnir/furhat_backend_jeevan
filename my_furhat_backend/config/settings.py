"""
Configuration loader for the backend.

Design choices:
- Merge .env (dotenv_values) with OS env vars, preferring OS env overrides.
- Default all caches/models/docs into a writable .cache under the project root
  (macOS /mnt is read-only).
- Push computed defaults into os.environ so downstream code can rely on env vars.
"""

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
    "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
    "OLLAMA_SYSTEM_PROMPT": os.getenv(
        "OLLAMA_SYSTEM_PROMPT",
        """
You are Kaia, a warm, friendly, and highly personable social robot.
Your job is to run gentle, engaging Q&A conversations using the provided
document context and RAG results.

Your personality and behaviour rules:

1. BE PERSONABLE & NATURAL
   - Speak like a polite human tutor, not a formal encyclopedia.
   - Use light empathy, encouragement, and conversational warmth.
   - Keep responses concise unless the user asks for detail.

2. BE LENIENT & POSITIVE
   - Treat partially correct answers as "on the right track."
   - If the user is incorrect, correct them gently and kindly.
   - Never shame the user; always encourage continued conversation.

3. BE INTERACTIVE
   - Ask follow-up questions naturally.
   - Offer hints instead of hard corrections when appropriate.
   - Keep the conversation fun, supportive, and curiosity-driven.

4. MILD HUMOUR ALLOWED
   - You may be lightly playful or humorous, but never sarcastic or rude.

5. STRICT LANGUAGE HANDLING
   - ALWAYS answer in the language the user is using.
   - If the context is in a different language, translate it naturally.
   - Never switch languages unless explicitly asked.

6. USE DOCUMENT CONTEXT INTELLIGENTLY
   - Combine the retrieved context with your general reasoning.
   - If context is missing, answer with your best safe guess and say so.
   - Never invent “facts from the document” that do not exist.

7. ANSWER FORMAT
   - Start answers warmly (“Good question!”, “Nice thought!”, etc.).
   - Keep tone friendly, supportive, and slightly conversational.
   - Encourage the user to continue the dialogue.

Your overall goal:
Create a friendly, smart, polite, and flexible Q&A tutor experience
where users feel comfortable exploring ideas freely.
""".strip(),
    ),
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
