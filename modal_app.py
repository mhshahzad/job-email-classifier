import modal

from email_inference import format_classification, format_email_text

MODEL_ID = "mhshahzad/distilbert-job-email-classifier_v3"
MODEL_REVISION = "88744c8694c667c524d9cf4cb878153a9085d5e6"
HF_SECRET_NAME = "huggingface-secret"

image = modal.Image.from_dockerfile("Dockerfile")

app = modal.App("job-email-classifier-private")


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    scaledown_window=300,
)
class JobEmailClassifier:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(torch.device("cpu"))
        self.model.eval()

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def classify(self, request: dict[str, object]) -> dict[str, object]:
        import torch
        import torch.nn.functional as F

        subject = request.get("subject")
        body = request.get("body")
        if not isinstance(subject, str) or not isinstance(body, str):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=422,
                detail="Request must include string fields 'subject' and 'body'.",
            )

        inputs = self.tokenizer(
            format_email_text(subject, body),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)[0].tolist()

        return format_classification(
            probabilities,
            self.model.config.id2label,
        )
