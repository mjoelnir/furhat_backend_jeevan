from typing import Dict

# Placeholder language ID to keep pipeline simple; can be swapped for a real
# langid model later (e.g., fasttext/langdetect) once dependencies are allowed.
def detect_language(text: str) -> str:
    """
    Placeholder: always returns 'en'.

    Chosen to keep dependencies minimal on the backend; language is primarily
    inferred client-side or via session distributions.
    """
    return "en"


def update_language_distribution(existing: Dict[str, float], lang_code: str, weight: float = 1.0) -> Dict[str, float]:
    """
    Update a language distribution dict with a new observation.
    Keeps probabilities normalized; lightweight alternative to heavier models.
    """
    dist = dict(existing) if existing else {}
    dist[lang_code] = dist.get(lang_code, 0.0) + weight
    total = sum(dist.values()) or 1.0
    for k in dist:
        dist[k] /= total
    return dist