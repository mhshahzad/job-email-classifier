from collections.abc import Mapping, Sequence
from typing import TypedDict


class ClassificationResult(TypedDict):
    predicted_category: str
    confidence: float
    all_scores: dict[str, float]


def format_email_text(subject: str, body: str) -> str:
    return f"Subject: {subject}\nEmail: {body}"


def format_classification(
    probabilities: Sequence[float],
    id2label: Mapping[int, str],
) -> ClassificationResult:
    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    predicted_label = id2label[predicted_id]

    return {
        "predicted_category": predicted_label,
        "confidence": round(float(probabilities[predicted_id]), 4),
        "all_scores": {
            id2label[index]: round(float(probability), 4)
            for index, probability in enumerate(probabilities)
        },
    }
