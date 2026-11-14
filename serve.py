import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

MODEL_ID = "mhshahzad/distilbert-job-email-classifier_v3"
MODEL_REVISION = "88744c8694c667c524d9cf4cb878153a9085d5e6"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN must be set to access the private Hugging Face model.")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token,
    )
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_path)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    app.state.model.to(torch.device("cpu"))
    app.state.model.eval()
    yield


app = FastAPI(title="Job Email Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: dict[str, Any]) -> dict[str, object]:
    subject = request.get("subject")
    body = request.get("body")
    if not isinstance(subject, str) or not isinstance(body, str):
        raise HTTPException(
            status_code=422,
            detail="Request must include string fields 'subject' and 'body'.",
        )

    import torch
    import torch.nn.functional as F

    from email_inference import format_classification, format_email_text

    inputs = app.state.tokenizer(
        format_email_text(subject, body),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = app.state.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)[0].tolist()

    return format_classification(
        probabilities,
        app.state.model.config.id2label,
    )
