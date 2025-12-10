# my_furhat_backend/perception/face.py
"""
Face recognition helpers using InsightFace.

Design choices:
- Load InsightFace once at import; if unavailable, fail soft (returns None) so
  the rest of the backend still works without GPU/onnxruntime.
- Use simple cosine similarity over stored embeddings; no external vector DB to
  keep dependencies light.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from my_furhat_backend.db.models import User

# --------- Load InsightFace model once at import ----------

try:
    from insightface.app import FaceAnalysis

    _face_app: Optional[FaceAnalysis] = FaceAnalysis(
        name="buffalo_l",  # solid general-purpose model
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f"[face] Failed to initialize InsightFace: {e}")
    _face_app = None


def _bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes into an OpenCV BGR array; fail soft on errors."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[face] Failed to decode image: {e}")
        return None


def extract_face_embedding(frame_bytes: bytes) -> Optional[np.ndarray]:
    """
    Extract a face embedding from an image.

    Returns:
        np.ndarray of shape (512,) or similar, or None if failed/no face.

    Rationale: keep it minimal—pick the largest face, use InsightFace normed
    embeddings; no batching, no multi-face disambiguation for now.
    """
    if _face_app is None:
        return None

    bgr = _bytes_to_bgr(frame_bytes)
    if bgr is None:
        return None

    faces = _face_app.get(bgr)
    if not faces:
        return None

    # For now, just take the largest detected face
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding  # already L2-normalized
    return np.array(emb, dtype=np.float32)


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Convert numpy embedding to raw float32 bytes."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Convert raw float32 bytes back to numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return -1.0
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def match_face_embedding(
    db: Session,
    embedding: np.ndarray,
    threshold: float = 0.45,
) -> Optional[Tuple[User, float]]:
    """
    Match a face embedding against stored users.

    Returns:
        (User, similarity) if similarity >= threshold, else None.

    Chosen approach: in-DB scan with cosine similarity—sufficient for small user
    sets; avoids adding a vector DB dependency.
    """
    if embedding is None or embedding.size == 0:
        return None

    users: List[User] = db.query(User).filter(User.face_embedding.isnot(None)).all()
    if not users:
        return None

    best_user = None
    best_score = -1.0

    for user in users:
        try:
            stored = deserialize_embedding(user.face_embedding)
            score = _cosine_similarity(embedding, stored)
            if score > best_score:
                best_score = score
                best_user = user
        except Exception as e:
            print(f"[face] Failed to compare embedding for user {user.id}: {e}")

    if best_user is not None and best_score >= threshold:
        return best_user, best_score

    return None