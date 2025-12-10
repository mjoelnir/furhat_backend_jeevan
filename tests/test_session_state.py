import pytest

from my_furhat_backend.perception import session_state


@pytest.fixture(autouse=True)
def reset_sessions():
    session_state._sessions.clear()  # type: ignore[attr-defined]
    yield
    session_state._sessions.clear()  # type: ignore[attr-defined]


def test_get_or_create_session_returns_same_instance():
    state_a = session_state.get_or_create_session("session-1")
    state_a.user_id = "user-123"

    state_b = session_state.get_or_create_session("session-1")

    assert state_a is state_b
    assert state_b.user_id == "user-123"


def test_update_session_user_assigns_and_returns_state():
    updated = session_state.update_session_user("session-42", "user-42")

    assert updated.user_id == "user-42"
    assert session_state.get_or_create_session("session-42").user_id == "user-42"


def test_update_session_language_dist_overwrites_previous_distribution():
    session_state.update_session_language_dist("session-2", {"en": 1.0})
    state = session_state.update_session_language_dist("session-2", {"en": 0.4, "no": 0.6})

    assert state.language_dist == {"en": 0.4, "no": 0.6}


def test_increment_turn_index_counts_from_one():
    assert session_state.increment_turn_index("session-turns") == 1
    assert session_state.increment_turn_index("session-turns") == 2

