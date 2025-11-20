"""
my_furhat_backend package initialization.

Provides compatibility shims required by downstream dependencies.
"""

from __future__ import annotations

try:
    import langgraph.checkpoint.base as _checkpoint_base

    if not hasattr(_checkpoint_base, "EXCLUDED_METADATA_KEYS"):
        _checkpoint_base.EXCLUDED_METADATA_KEYS = set()  # type: ignore[attr-defined]
except Exception:
    # LangGraph is optional during docs/tests; ignore if unavailable.
    pass
