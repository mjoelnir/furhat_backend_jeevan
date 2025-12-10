import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from my_furhat_backend.perception import session_state
from my_furhat_backend.perception.websocket_handler import (
    AUDIO_PREFIX,
    VIDEO_PREFIX,
    _generate_new_user_id,
    _handle_binary_message,
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


def test_handle_binary_message_video_sends_identity_update():
    """
    Simulate a single video frame and verify that an identity_update
    payload is pushed back over the websocket.
    """
    ws = DummyWebSocket()
    db = MagicMock()
    session_id = "session-video"

    # Ensure session exists
    session_state.get_or_create_session(session_id)

    dummy_user = SimpleNamespace(id="user-vid", name="Ava", languages_json={"en": 1.0})

    # Patch the low-level face handling so we don't depend on real models
    with patch(
        "my_furhat_backend.perception.websocket_handler._update_identity_from_face",
        return_value=dummy_user,
    ):
        frame_bytes = b"\x00\x01\x02\x03"
        data = bytes([VIDEO_PREFIX]) + frame_bytes
        asyncio.run(
            _handle_binary_message(
                ws,
                db,
                data=data,
                current_session_id=session_id,
            )
        )

    assert ws.sent_text, "Expected identity_update after video frame"
    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "identity_update"
    assert payload["payload"]["user_id"] == "user-vid"


def test_handle_binary_message_audio_sends_identity_update():
    """
    Simulate a single audio chunk and verify that an identity_update
    payload is pushed back over the websocket.
    """
    ws = DummyWebSocket()
    db = MagicMock()
    session_id = "session-audio"

    # Ensure session exists
    session_state.get_or_create_session(session_id)

    dummy_user = SimpleNamespace(id="user-aud", name="Ava", languages_json={"en": 1.0})

    with patch(
        "my_furhat_backend.perception.websocket_handler._update_identity_from_voice",
        return_value=dummy_user,
    ):
        audio_bytes = b"\x10\x20\x30\x40"
        data = bytes([AUDIO_PREFIX]) + audio_bytes
        asyncio.run(
            _handle_binary_message(
                ws,
                db,
                data=data,
                current_session_id=session_id,
            )
        )

    assert ws.sent_text, "Expected identity_update after audio chunk"
    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "identity_update"
    assert payload["payload"]["user_id"] == "user-aud"

