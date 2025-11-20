from typing import Dict


def detect_language(text: str) -> str:
    """
    VERY simple placeholder – always returns 'en' for now.
    Later: plug in a real language-id model or reuse ASR metadata.
    """
    return "en"


def update_language_distribution(existing: Dict[str, float], lang_code: str, weight: float = 1.0) -> Dict[str, float]:
    """
    Update a language distribution dict with a new observation.
    existing: e.g. {"en": 0.7, "no": 0.3}
    lang_code: e.g. "en"
    """
    dist = dict(existing) if existing else {}
    dist[lang_code] = dist.get(lang_code, 0.0) + weight
    total = sum(dist.values()) or 1.0
    for k in dist:
        dist[k] /= total
    return dist