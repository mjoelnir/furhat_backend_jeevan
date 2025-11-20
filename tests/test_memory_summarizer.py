from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from my_furhat_backend.memory import summarizer


@dataclass
class DummyTurn:
    user_text: str | None


def test_naive_summary_empty_turns():
    assert summarizer._naive_summary([]) == ""


def test_naive_summary_uses_first_and_last_turns():
    turns = [
        DummyTurn("Hello there"),
        DummyTurn("Some middle question"),
        DummyTurn("Thanks, bye"),
    ]

    summary = summarizer._naive_summary(turns)

    assert "Hello there" in summary
    assert "Thanks, bye" in summary
    assert "Some middle" not in summary


def test_update_summary_for_conversation_calls_crud():
    db = MagicMock()
    conversation = MagicMock()
    turn = DummyTurn("Hi")

    with patch.object(summarizer.crud, "get_recent_turns", return_value=[turn]) as get_recent, patch.object(
        summarizer.crud, "update_conversation_summary"
    ) as update_summary:
        summarizer.update_summary_for_conversation(db, conversation)

    get_recent.assert_called_once_with(db, conversation, limit=50)
    update_summary.assert_called_once()
    args, kwargs = update_summary.call_args
    assert args[0] is db
    assert args[1] is conversation

