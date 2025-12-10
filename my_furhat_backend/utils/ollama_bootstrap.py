"""
Utility helpers to bootstrap the local Ollama runtime on backend startup.

Design choice: keep startup resilient and dependency-light.
- Prefer to reuse an already running Ollama instance; only try to start it if unreachable.
- Pull the configured model once per process to avoid repeated downloads.
- Avoid tight coupling: we shell out to `ollama` binary instead of embedding the server.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Optional

import requests

from my_furhat_backend.config.settings import config

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = config.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = config.get("OLLAMA_MODEL", "llama3.2:latest")

_bootstrap_attempted = False


def ensure_ollama_ready(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """
    Ensure an Ollama server is reachable and the requested model is pulled.

    Why this approach:
    - Skip work if already attempted (idempotent per process).
    - Best-effort: continue without Ollama if binary/server is missing, so the
      rest of the backend can still serve non-LLM endpoints.
    """
    global _bootstrap_attempted
    if _bootstrap_attempted:
        return
    _bootstrap_attempted = True

    model = model or DEFAULT_MODEL
    base_url = base_url or DEFAULT_BASE_URL

    if not model:
        logger.info("OLLAMA_MODEL not configured; skipping Ollama bootstrap.")
        return

    if not _ping_server(base_url):
        if not _start_server():
            logger.warning("Could not start Ollama server; continuing without it.")
            return

    if not _ping_server(base_url, timeout=3):
        logger.warning("Ollama server still unreachable at %s.", base_url)
        return

    _pull_model(model)


def _ping_server(base_url: str, timeout: float = 2.0) -> bool:
    try:
        requests.get(f"{base_url}/api/tags", timeout=timeout).raise_for_status()
        return True
    except Exception:
        return False


def _start_server() -> bool:
    ollama_binary = shutil.which("ollama")
    if not ollama_binary:
        logger.warning("Cannot start Ollama server because 'ollama' binary is missing.")
        return False

    logger.info("Starting Ollama server via '%s serve'.", ollama_binary)
    try:
        subprocess.Popen(
            [ollama_binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("Failed to launch Ollama server: %s", exc)
        return False

    for _ in range(10):
        time.sleep(1)
        if _ping_server(DEFAULT_BASE_URL):
            logger.info("Ollama server is now reachable.")
            return True

    logger.warning("Timed out waiting for Ollama server to become reachable.")
    return False


def _pull_model(model: str) -> None:
    ollama_binary = shutil.which("ollama")
    if not ollama_binary:
        logger.warning("Cannot pull Ollama model because 'ollama' binary is missing.")
        return

    logger.info("Ensuring Ollama model '%s' is pulled.", model)
    try:
        subprocess.run(
            [ollama_binary, "pull", model],
            check=True,
        )
        logger.info("Model '%s' is available locally.", model)
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to pull Ollama model '%s': %s", model, exc)

