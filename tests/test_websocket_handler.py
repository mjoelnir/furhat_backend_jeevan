import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from my_furhat_backend.perception import session_state
from my_furhat_backend.perception.websocket_handler import (
    _handle_text_message,
    _send_error,
)


class DummyWebSocket:
    def __init__(self):
        self.sent_text = []

    async def send_text(self, message: str):
        self.sent_text.append(message)


@pytest.fixture(autouse=True)
def reset_sessions():
    session_state._sessions.clear()  # type: ignore[attr-defined]
    yield
    session_state._sessions.clear()  # type: ignore[attr-defined]


def test_send_error_wraps_payload():
    ws = DummyWebSocket()

    asyncio.run(_send_error(ws, "session-err", "Something went wrong"))

    assert ws.sent_text, "Expected error payload to be sent"
    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "error"
    assert payload["payload"]["session_id"] == "session-err"
    assert payload["payload"]["message"] == "Something went wrong"


def test_handle_text_message_turn_updates_language_distribution():
    ws = DummyWebSocket()
    db = MagicMock()
    session_id = "session-turn"
    user = SimpleNamespace(id="user-1", name="Ava", languages_json={"en": 1.0})

    hello = json.dumps({"type": "hello", "payload": {"session_id": session_id}})
    asyncio.run(_handle_text_message(ws, db, hello, None))

    with patch(
        "my_furhat_backend.perception.websocket_handler.crud.get_user_by_id",
        return_value=user,
    ), patch(
        "my_furhat_backend.perception.websocket_handler.crud.get_or_create_conversation",
        return_value=MagicMock(),
    ) as get_conv, patch(
        "my_furhat_backend.perception.websocket_handler.crud.create_turn"
    ) as create_turn, patch(
        "my_furhat_backend.perception.websocket_handler.crud.update_user_languages"
    ) as update_lang:
        turn_payload = json.dumps(
            {
                "type": "turn",
                "payload": {
                    "session_id": session_id,
                    "user_text": "Hei der",
                    "language": "no",
                    "robot_text": "Hei! Hvordan går det?",
                },
            }
        )

        asyncio.run(_handle_text_message(ws, db, turn_payload, session_id))

    state = session_state.get_or_create_session(session_id)
    assert state.language_dist.get("no", 0) > 0
    assert "no" in state.language_dist


def test_handle_text_message_name_update_updates_user(monkeypatch):
    ws = DummyWebSocket()
    db = MagicMock()
    session_id = "session-name"
    user = SimpleNamespace(id="user-1", name=None, languages_json={"en": 1.0})

    asyncio.run(
        _handle_text_message(
            ws,
            db,
            json.dumps({"type": "hello", "payload": {"session_id": session_id}}),
            None,
        )
    )

    with patch(
        "my_furhat_backend.perception.websocket_handler.crud.get_user_by_id",
        return_value=user,
    ), patch(
        "my_furhat_backend.perception.websocket_handler.crud.update_user_name",
        side_effect=lambda _db, existing, new_name: setattr(existing, "name", new_name),
    ) as update_name:
        name_payload = json.dumps(
            {
                "type": "name_update",
                "payload": {"session_id": session_id, "name": "Ava"},
            }
        )
        asyncio.run(_handle_text_message(ws, db, name_payload, session_id))

    update_name.assert_called_once()
    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "identity_update"
    assert payload["payload"]["name"] == "Ava"


def test_handle_text_message_unknown_type_sends_error():
    ws = DummyWebSocket()
    db = MagicMock()
    session_id = "session-unknown"

    asyncio.run(
        _handle_text_message(
            ws,
            db,
            json.dumps({"type": "hello", "payload": {"session_id": session_id}}),
            None,
        )
    )

    asyncio.run(
        _handle_text_message(
            ws,
            db,
            json.dumps({"type": "foobar", "payload": {"session_id": session_id}}),
            session_id,
        )
    )

    payload = json.loads(ws.sent_text[-1])
    assert payload["type"] == "error"
    assert "Unknown message type" in payload["payload"]["message"]

