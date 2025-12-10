from sqlalchemy.orm import Session
from my_furhat_backend.db import crud
from my_furhat_backend.db.models import Conversation


def _naive_summary(turns):
    """
    Very simple summary: first + last user utterances trimmed.

    Design choice: keep it cheap and dependency-free. This is a placeholder
    until an LLM-based summarizer is plugged in; avoids extra calls when not
    needed but keeps conversation summaries non-empty.
    """
    if not turns:
        return ""
    first = (turns[0].user_text or "").strip()
    last = (turns[-1].user_text or "").strip()
    if not first and not last:
        return ""
    if first == last:
        return first[:200]
    return f"Started with: {first[:100]}... | Ended with: {last[:100]}..."


def update_summary_for_conversation(db: Session, conversation: Conversation):
    """
    Compute and persist a conversation summary using the naive summarizer.

    Rationale: touch only the last N turns (limit=50) to keep it inexpensive
    on each call; swap _naive_summary with an LLM call when desired.
    """
    turns = crud.get_recent_turns(db, conversation, limit=50)
    summary = _naive_summary(turns)
    crud.update_conversation_summary(db, conversation, summary)