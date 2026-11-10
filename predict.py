
# predict.py

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "models/distilbert-job-classifier"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()


def classify_email(subject, body):

    text = (
        f"Subject: {subject}\n"
        f"Email: {body}"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_id = int(
        torch.argmax(probabilities)
    )

    predicted_label = model.config.id2label[
        predicted_id
    ]

    scores = {
        model.config.id2label[i]: round(
            float(probabilities[i]), 4
        )
        for i in range(len(probabilities))
    }

    return {
        "predicted_category": predicted_label,
        "confidence": round(
            float(probabilities[predicted_id]), 4
        ),
        "all_scores": scores
    }


if __name__ == "__main__":

    result = classify_email(
        subject="Interview Invitation - Data Analyst",
        body=(
            "Thank you for applying. "
            "We would like to invite you for an "
            "interview next week."
        )
    )

    print("\nClassification result:")
    print(result)
