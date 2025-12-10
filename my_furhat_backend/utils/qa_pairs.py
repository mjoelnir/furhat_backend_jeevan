"""
Utility helpers for loading question–answer pairs from a JSON file.

Design choice: simple JSON + in-memory cache to keep startup fast and avoid
additional dependencies. QA pairs are static and small, so no DB/index needed.

Expected format:
{
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from my_furhat_backend.config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    index: int
    question: str
    answer: str


_CACHE: List[QAPair] | None = None


def _qa_json_path() -> Path:
    """
    Resolve the path to qa_pairs.json using DOCUMENTS_PATH from settings.
    Keeping it relative to config allows swapping datasets without code changes.
    """
    base = Path(config["DOCUMENTS_PATH"])
    return base / "qa_pairs.json"


def load_qa_pairs(force_reload: bool = False) -> List[QAPair]:
    """
    Load all QA pairs from qa_pairs.json with in-memory caching.

    Chosen approach: simple lazy cache in-process; avoids repeated disk I/O and
    keeps runtime dependencies minimal (no DB/FS watchers). Use force_reload=True
    if the file changes while the process is alive.
    """
    global _CACHE

    if _CACHE is not None and not force_reload:
        return _CACHE

    path = _qa_json_path()
    if not path.is_file():
        logger.warning("qa_pairs.json not found at %s", path)
        _CACHE = []
        return _CACHE

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load qa_pairs.json from %s: %s", path, exc)
        _CACHE = []
        return _CACHE

    pairs = data.get("qa_pairs")
    if not isinstance(pairs, list):
        logger.warning("qa_pairs.json at %s does not contain a 'qa_pairs' list", path)
        _CACHE = []
        return _CACHE

    result: List[QAPair] = []
    for idx, item in enumerate(pairs):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        result.append(QAPair(index=idx, question=question, answer=answer))

    _CACHE = result
    logger.info("Loaded %d QA pairs from %s", len(_CACHE), path)
    return _CACHE


def get_qa_pair(index: int) -> Optional[QAPair]:
    pairs = load_qa_pairs()
    if 0 <= index < len(pairs):
        return pairs[index]
    return None


def get_random_qa_pair() -> Optional[QAPair]:
    import random

    pairs = load_qa_pairs()
    if not pairs:
        return None
    return random.choice(pairs)



