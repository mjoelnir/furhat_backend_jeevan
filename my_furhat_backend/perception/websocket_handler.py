from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from my_furhat_backend.db.session import SessionLocal
from my_furhat_backend.db import crud
from my_furhat_backend.perception.session_state import (
    get_or_create_session,
    update_session_user,
    update_session_language_dist,
    increment_turn_index,
)
from my_furhat_backend.perception.language import detect_language, update_language_distribution
from my_furhat_backend.perception import face as face_mod
from my_furhat_backend.perception import voice as voice_mod


VIDEO_PREFIX = 0x01
AUDIO_PREFIX = 0x02


def _generate_new_user_id() -> str:
    return f"user-{uuid.uuid4().hex[:12]}"


async def _send_error(websocket: WebSocket, session_id: str, message: str):
    await websocket.send_text(
        json.dumps(
            {
                "type": "error",
                "payload": {
                    "session_id": session_id,
                    "code": "BAD_REQUEST",
                    "message": message,
                },
            }
        )
    )


async def _send_identity_update(
    websocket: WebSocket,
    session_id: str,
    user_id: str,
    name: Optional[str],
    languages_dist: Dict[str, float],
):
    primary_language = max(languages_dist, key=languages_dist.get) if languages_dist else "en"
    payload = {
        "type": "identity_update",
        "payload": {
            "session_id": session_id,
            "user_id": user_id,
            "name": name,
            "primary_language": primary_language,
            "languages": languages_dist,
            "confidence": float(languages_dist.get(primary_language, 1.0)),
            "last_seen": int(datetime.utcnow().timestamp() * 1000),
        },
    }
    await websocket.send_text(json.dumps(payload))


def _ensure_user_for_session(db: Session, session_id: str):
    """
    Ensure there is a User associated with this session.
    """
    state = get_or_create_session(session_id)
    if state.user_id:
        user = crud.get_user_by_id(db, state.user_id)
        if user:
            return user

    # No user yet → create one
    user_id = _generate_new_user_id()
    user = crud.create_user(
        db,
        user_id=user_id,
        primary_language="en",
        languages_json={"en": 1.0},
    )
    update_session_user(session_id, user_id)
    update_session_language_dist(session_id, user.languages_json or {"en": 1.0})
    return user


def _update_identity_from_face(db: Session, session_id: str, frame_bytes: bytes):
    state = get_or_create_session(session_id)

    emb = face_mod.extract_face_embedding(frame_bytes)
    if emb is None:
        return None

    match = face_mod.match_face_embedding(db, emb)
    serialized = face_mod.serialize_embedding(emb)

    if match:
        user, score = match
        update_session_user(session_id, user.id)
        crud.update_user_embeddings(db, user, face_embedding=serialized)
        return user

    user = _ensure_user_for_session(db, session_id)
    crud.update_user_embeddings(db, user, face_embedding=serialized)
    return user


def _update_identity_from_voice(db: Session, session_id: str, audio_bytes: bytes):
    state = get_or_create_session(session_id)

    emb = voice_mod.extract_voice_embedding(audio_bytes)
    if emb is None:
        return None

    match = voice_mod.match_voice_embedding(db, emb)
    serialized = voice_mod.serialize_embedding(emb)

    if match:
        user, score = match
        update_session_user(session_id, user.id)
        crud.update_user_embeddings(db, user, voice_embedding=serialized)
        return user

    user = _ensure_user_for_session(db, session_id)
    crud.update_user_embeddings(db, user, voice_embedding=serialized)
    return user


async def _handle_text_message(
    websocket: WebSocket,
    db: Session,
    text: str,
    current_session_id: Optional[str],
) -> Optional[str]:
    """
    Handle JSON text messages: hello, turn, name_update.
    Returns updated current_session_id.
    """
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        await _send_error(websocket, current_session_id or "unknown", "Invalid JSON")
        return current_session_id

    msg_type = msg.get("type")
    payload: Dict[str, Any] = msg.get("payload") or {}

    # --- HELLO ---
    if msg_type == "hello":
        session_id = payload.get("session_id")
        if not session_id:
            await _send_error(websocket, "unknown", "Missing session_id in hello payload")
            return current_session_id

        get_or_create_session(session_id)
        # Default language distribution
        update_session_language_dist(session_id, {"en": 1.0})
        return session_id

    # For all other types, we need session_id
    session_id = payload.get("session_id") or current_session_id
    if not session_id:
        await _send_error(websocket, "unknown", f"Missing session_id in {msg_type} payload")
        return current_session_id

    # Make sure session exists
    state = get_or_create_session(session_id)

    # --- TURN ---
    if msg_type == "turn":
        # If we already have a recognized user, reuse it; else create.
        if state.user_id:
            user = crud.get_user_by_id(db, state.user_id)
            if not user:
                user = _ensure_user_for_session(db, session_id)
        else:
            user = _ensure_user_for_session(db, session_id)

        user_text = payload.get("user_text") or ""
        lang = payload.get("language") or detect_language(user_text or "")

        # Update language distribution
        new_lang_dist = update_language_distribution(state.language_dist, lang)
        update_session_language_dist(session_id, new_lang_dist)

        crud.update_user_languages(
            db,
            user,
            new_lang_dist,
            primary_language=max(new_lang_dist, key=new_lang_dist.get),
        )

        # Create conversation + turn
        conv = crud.get_or_create_conversation(
            db,
            session_id=session_id,
            user=user,
            language=lang,
        )
        turn_idx = increment_turn_index(session_id)
        crud.create_turn(
            db,
            conversation=conv,
            turn_index=turn_idx,
            user_text=user_text,
            robot_text=payload.get("robot_text"),
        )

        await _send_identity_update(
            websocket,
            session_id=session_id,
            user_id=user.id,
            name=user.name,
            languages_dist=new_lang_dist,
        )

    # --- NAME_UPDATE ---
    elif msg_type == "name_update":
        if state.user_id:
            user = crud.get_user_by_id(db, state.user_id)
            if not user:
                user = _ensure_user_for_session(db, session_id)
        else:
            user = _ensure_user_for_session(db, session_id)

        name = payload.get("name")
        if name:
            crud.update_user_name(db, user, name)

        lang_dist = state.language_dist or user.languages_json or {"en": 1.0}
        update_session_language_dist(session_id, lang_dist)

        await _send_identity_update(
            websocket,
            session_id=session_id,
            user_id=user.id,
            name=user.name,
            languages_dist=lang_dist,
        )

    else:
        await _send_error(websocket, session_id, f"Unknown message type: {msg_type}")

    return session_id


async def _handle_binary_message(
    websocket: WebSocket,
    db: Session,
    data: bytes,
    current_session_id: Optional[str],
):
    """
    Handle binary streaming frames:
      - 0x01 + JPEG bytes => video frame
      - 0x02 + audio bytes => audio chunk
    """
    if not data:
        return

    stream_type = data[0]
    payload_bytes = data[1:]

    if not current_session_id:
        # We require a hello first to set session_id
        await _send_error(websocket, "unknown", "Binary data received before hello/session_id")
        return

    session_id = current_session_id
    state = get_or_create_session(session_id)

    if stream_type == VIDEO_PREFIX:
        user = _update_identity_from_face(db, session_id, payload_bytes)
    elif stream_type == AUDIO_PREFIX:
        user = _update_identity_from_voice(db, session_id, payload_bytes)
    else:
        # Unknown binary subtype; ignore
        return

    if not user:
        return

    # Use existing or default language distribution
    lang_dist = state.language_dist or user.languages_json or {"en": 1.0}
    update_session_language_dist(session_id, lang_dist)

    await _send_identity_update(
        websocket,
        session_id=session_id,
        user_id=user.id,
        name=user.name,
        languages_dist=lang_dist,
    )


async def perception_ws_handler(websocket: WebSocket):
    """
    Single WebSocket entry point that handles:
      - Text JSON messages for hello/turn/name_update.
      - Binary streaming frames from the Furhat camera/mic.

    Designed to run continuously while the skill is active.
    """
    await websocket.accept()
    db: Session = SessionLocal()
    current_session_id: Optional[str] = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                current_session_id = await _handle_text_message(
                    websocket,
                    db,
                    message["text"],
                    current_session_id,
                )

            elif "bytes" in message and message["bytes"] is not None:
                await _handle_binary_message(
                    websocket,
                    db,
                    message["bytes"],
                    current_session_id,
                )

            # Otherwise ignore (e.g. pings)
    except WebSocketDisconnect:
        pass
    finally:
        db.close()