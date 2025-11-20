import asyncio
import json
import pytest

from my_furhat_backend.perception import session_state
from my_furhat_backend.perception.websocket_handler import (
    _generate_new_user_id,
    _handle_text_message,
    _send_identity_update,
)


@pytest.fixture(autouse=True)
def reset_session_state():
    session_state._sessions.clear()  # type: ignore[attr-defined]
    yield
    session_state._sessions.clear()  # type: ignore[attr-defined]


class DummyWebSocket:
    def __init__(self):
        self.sent_text: list[str] = []

    async def send_text(self, text: str):
        self.sent_text.append(text)


def test_generate_new_user_id_produces_unique_ids():
    ids = {_generate_new_user_id() for _ in range(10)}

    assert all(identifier.startswith("user-") for identifier in ids)
    assert len(ids) == 10


def test_send_identity_update_formats_payload():
    ws = DummyWebSocket()

    asyncio.run(
        _send_identity_update(
            ws,
            session_id="session-123",
            user_id="user-xyz",
            name="Ava",
            languages_dist={"en": 0.7, "no": 0.3},
        )
    )

    assert len(ws.sent_text) == 1
    payload = json.loads(ws.sent_text[0])
    assert payload["type"] == "identity_update"
    assert payload["payload"]["session_id"] == "session-123"
    assert payload["payload"]["user_id"] == "user-xyz"
    assert payload["payload"]["primary_language"] == "en"


def test_handle_text_message_initializes_session_language_distribution():
    ws = DummyWebSocket()
    text = json.dumps(
        {
            "type": "hello",
            "payload": {"session_id": "session-abc"},
        }
    )

    session_id = asyncio.run(_handle_text_message(ws, db=None, text=text, current_session_id=None))

    assert session_id == "session-abc"
    state = session_state.get_or_create_session("session-abc")
    assert state.language_dist == {"en": 1.0}


def test_handle_text_message_without_session_sends_error():
    ws = DummyWebSocket()
    text = json.dumps({"type": "hello", "payload": {}})

    asyncio.run(_handle_text_message(ws, db=None, text=text, current_session_id=None))

    assert ws.sent_text, "Expected an error payload to be sent"
    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "error"

