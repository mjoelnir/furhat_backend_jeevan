try:
    from transformers import pipeline  # type: ignore[import]
except Exception as e:  # noqa: BLE001
    print(
        f"[classifier] Transformers pipeline unavailable, falling back to "
        f"naive string-similarity classifier: {e}"
    )
    pipeline = None  # type: ignore[assignment]


class TextClassifier:
    """
    Text classifier for ranking documents against a query.

    Design:
    - Preferred: Hugging Face zero-shot classification (if transformers+torch available).
    - Fallback: simple lexical overlap scoring when HF stack is not available, to avoid hard
      dependency on GPU/libtorch for small doc sets.
    """

    def __init__(self, model_id: str = "facebook/bart-large-mnli"):
        """
        Initialize the TextClassifier.

        If transformers.pipeline is available, create a zero-shot classification
        pipeline. Otherwise, use a lightweight lexical heuristic to keep running
        without the HF stack.
        """
        if pipeline is None:
            self.classifier = None
        else:
            self.classifier = pipeline("zero-shot-classification", model=model_id)

    def _lexical_score(self, text: str, label: str) -> float:
        """
        Very simple similarity: proportion of label tokens that appear in the text.
        """
        text_l = text.lower()
        label_tokens = [tok for tok in label.lower().split() if tok]
        if not label_tokens:
            return 0.0
        matches = sum(1 for tok in label_tokens if tok in text_l)
        return matches / len(label_tokens)

    def classify(self, text: str, labels: list[str]) -> dict:
        """
        Rank labels for a given text.

        Returns:
            dict: {label: score}, sorted by score descending.
        """
        if not labels:
            return {}

        # HF-based zero-shot classification if available
        if self.classifier is not None:
            scores = self.classifier(text, labels, multi_label=True)
            scores_with_labels = dict(zip(scores["labels"], scores["scores"]))
        else:
            # Fallback: lexical overlap
            scores_with_labels = {label: self._lexical_score(text, label) for label in labels}

        # Sort the dictionary by scores in descending order.
        sorted_items = sorted(scores_with_labels.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_items)
