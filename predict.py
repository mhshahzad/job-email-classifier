
# predict.py

import torch
import torch.nn.functional as F

from email_inference import format_classification, format_email_text
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

    text = format_email_text(subject, body)

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

    return format_classification(
        probabilities.tolist(),
        model.config.id2label,
    )


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
