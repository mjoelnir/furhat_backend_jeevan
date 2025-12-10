from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from my_furhat_backend.perception import face, voice


def _make_face_user(user_id: str, vector: np.ndarray):
    return SimpleNamespace(
        id=user_id,
        face_embedding=face.serialize_embedding(vector.astype(np.float32)),
    )


def _make_voice_user(user_id: str, vector: np.ndarray):
    return SimpleNamespace(
        id=user_id,
        voice_embedding=voice.serialize_embedding(vector.astype(np.float32)),
    )


def test_match_face_embedding_returns_best_user():
    db = MagicMock()
    query = db.query.return_value
    filtered = query.filter.return_value

    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    users = [
        _make_face_user("user-a", np.array([0.9, 0.0, 0.0])),
        _make_face_user("user-b", np.array([0.0, 1.0, 0.0])),
    ]
    filtered.all.return_value = users

    match = face.match_face_embedding(db, target, threshold=0.4)

    assert match is not None
    user, score = match
    assert user.id == "user-a"
    assert score > 0.8


def test_match_voice_embedding_returns_none_without_users():
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    result = voice.match_voice_embedding(db, np.array([1.0, 0.0], dtype=np.float32))

    assert result is None


def test_extract_voice_embedding_returns_none_without_encoder(monkeypatch):
    monkeypatch.setattr(voice, "_voice_encoder", None)

    assert voice.extract_voice_embedding(b"fake-bytes") is None

