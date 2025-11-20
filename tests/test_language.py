import pytest

from my_furhat_backend.perception.language import (
    detect_language,
    update_language_distribution,
)


def test_detect_language_returns_default_en():
    assert detect_language("Hei på deg!") == "en"
    assert detect_language("") == "en"


def test_update_language_distribution_normalizes_scores():
    existing = {"en": 2.0, "no": 1.0}

    updated = update_language_distribution(existing, "no", weight=3.0)

    assert pytest.approx(sum(updated.values()), rel=1e-6) == 1.0
    assert updated["no"] > updated["en"]


def test_update_language_distribution_handles_empty_seed():
    updated = update_language_distribution({}, "en")

    assert updated == {"en": 1.0}

