FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /root

RUN python -m pip install --no-cache-dir \
    "fastapi[standard]" \
    huggingface_hub==2.4.0 \
    torch==2.15.0 \
    transformers==5.22.2

COPY email_inference.py /root/email_inference.py
COPY serve.py /root/serve.py

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
