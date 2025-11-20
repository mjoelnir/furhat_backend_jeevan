# my_furhat_backend/perception/voice.py

from __future__ import annotations

import io
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from sqlalchemy.orm import Session

from my_furhat_backend.db.models import User

# --------- Load Resemblyzer encoder once ----------

try:
    from resemblyzer import VoiceEncoder, preprocess_wav

    _voice_encoder: Optional[VoiceEncoder] = VoiceEncoder()
except Exception as e:
    print(f"[voice] Failed to initialize VoiceEncoder: {e}")
    _voice_encoder = None


def _bytes_to_mono_float32(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Decode audio bytes into a mono float32 numpy array."""
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:  # stereo → mono
            audio = np.mean(audio, axis=1)
        return preprocess_wav(audio, source_sr=sr)
    except Exception as e:
        print(f"[voice] Failed to decode audio: {e}")
        return None


def extract_voice_embedding(audio_bytes: bytes) -> Optional[np.ndarray]:
    """
    Extract a voice embedding from an audio clip (a few seconds).

    Returns:
        np.ndarray or None.
    """
    if _voice_encoder is None:
        return None

    wav = _bytes_to_mono_float32(audio_bytes)
    if wav is None or len(wav) < 16000:  # < 1 s at 16k
        return None

    emb = _voice_encoder.embed_utterance(wav)
    return np.array(emb, dtype=np.float32)


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return -1.0
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def match_voice_embedding(
    db: Session,
    embedding: np.ndarray,
    threshold: float = 0.75,
) -> Optional[Tuple[User, float]]:
    """
    Match a voice embedding against stored users.

    Returns:
        (User, similarity) if similarity >= threshold, else None.
    """
    if embedding is None or embedding.size == 0:
        return None

    users: List[User] = db.query(User).filter(User.voice_embedding.isnot(None)).all()
    if not users:
        return None

    best_user = None
    best_score = -1.0

    for user in users:
        try:
            stored = deserialize_embedding(user.voice_embedding)
            score = _cosine_similarity(embedding, stored)
            if score > best_score:
                best_score = score
                best_user = user
        except Exception as e:
            print(f"[voice] Failed to compare embedding for user {user.id}: {e}")

    if best_user is not None and best_score >= threshold:
        return best_user, best_score

    return None