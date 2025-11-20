from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class SessionState:
    session_id: str
    user_id: Optional[str] = None
    # language distribution ("en" -> prob, etc.)
    language_dist: Dict[str, float] = field(default_factory=dict)
    turn_index: int = 0


# Simple in-memory store: session_id -> SessionState
_sessions: Dict[str, SessionState] = {}


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id=session_id)
    return _sessions[session_id]


def update_session_user(session_id: str, user_id: str) -> SessionState:
    state = get_or_create_session(session_id)
    state.user_id = user_id
    return state


def update_session_language_dist(session_id: str, language_dist: Dict[str, float]) -> SessionState:
    state = get_or_create_session(session_id)
    state.language_dist = language_dist
    return state


def increment_turn_index(session_id: str) -> int:
    state = get_or_create_session(session_id)
    state.turn_index += 1
    return state.turn_index