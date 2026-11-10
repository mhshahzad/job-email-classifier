
# train.py

import os
import re
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from huggingface_hub import HfApi

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = os.getenv("HF_MODEL_REPO")

BASE_MODEL = "distilbert-base-uncased"

DATASET_DIR = Path(__file__).parent / "training_dataset"
OUTPUT_DIR = Path("models/distilbert-job-classifier")
EVALUATION_DIR = Path("evaluation")

for directory in [OUTPUT_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5

PUSH_TO_HUB = bool(MODEL_REPO)

LABELS = [
    "application_confirmation",
    "assessment",
    "interview_invitation",
    "offer",
    "rejection"
]

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD LOCAL DATASET SPLITS
# ============================================================

def clean_text(text):
    text = str(text or "")

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def load_split(name):
    path = DATASET_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Local {name} split not found: {path}")

    split_df = pd.read_json(path, lines=True)
    required_fields = {"subject", "body", "label"}
    missing_fields = required_fields - set(split_df.columns)
    if missing_fields:
        raise ValueError(
            f"{path} is missing required fields: {', '.join(sorted(missing_fields))}"
        )

    split_df = split_df[["subject", "body", "label"]].copy()
    for field in ("subject", "body"):
        split_df[field] = split_df[field].fillna("").astype(str).apply(clean_text)
    split_df["label"] = split_df["label"].astype(str).str.strip().str.lower()

    invalid_labels = sorted(set(split_df["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(
            f"{path} contains unsupported labels: {', '.join(invalid_labels)}"
        )

    split_df["text"] = (
        "Subject: " + split_df["subject"] + "\nEmail: " + split_df["body"]
    )
    split_df = split_df[
        split_df["text"].str.strip().ne("Subject: \nEmail:")
    ].copy()
    split_df = split_df.drop_duplicates(subset=["text", "label"]).reset_index(
        drop=True
    )

    if split_df.empty:
        raise ValueError(f"No usable records found in {path}")
    return split_df


print(f"\nLoading local dataset splits from {DATASET_DIR}...")
train_df = load_split("train")
validation_df = load_split("validation")
test_df = load_split("test")

for name, split_df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df),
]:
    print(f"\n{name}: {len(split_df)} records")
    print(split_df["label"].value_counts().sort_index())

missing_train_labels = set(LABELS) - set(train_df["label"])
if missing_train_labels:
    raise ValueError(
        "Training split is missing expected labels: "
        + ", ".join(sorted(missing_train_labels))
    )
if train_df["label"].value_counts().min() < 3:
    raise ValueError(
        "Training split must contain at least three examples for every label."
    )


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

def prepare_split(split_df):

    result = split_df[["text", "label"]].copy()

    result["label"] = result["label"].map(
        LABEL2ID
    )

    result = result.rename(
        columns={"label": "labels"}
    )

    return Dataset.from_pandas(
        result,
        preserve_index=False
    )


dataset = DatasetDict({
    "train": prepare_split(train_df),
    "validation": prepare_split(validation_df),
    "test": prepare_split(test_df)
})


# ============================================================
# TOKENIZATION
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_labels = np.array(
    train_df["label"].map(LABEL2ID)
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(LABELS)),
    y=train_labels
)

# Prevent extreme weights for very rare classes.
weights = np.clip(weights, 0, 10)

# Normalize weights.
weights = weights / weights.mean()

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
)

print("Class weights:")

for label, weight in zip(LABELS, weights):
    print(f"{label}: {weight:.4f}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading DistilBERT classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    logits, true_labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=np.arange(len(LABELS)),
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1))
    }


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs["labels"]

        outputs = model(**inputs)

        loss_function = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                outputs.logits.device
            )
        )

        loss = loss_function(
            outputs.logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=0.10,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,
    data_seed=SEED,

    dataloader_num_workers=0
)


# ============================================================
# TRAIN
# ============================================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics,

    class_weights=class_weights,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

print("\nStarting DistilBERT training...")

trainer.train()

# Save the best selected model.
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on untouched test dataset...")

inference_started = time.perf_counter()
test_results = trainer.predict(
    tokenized_dataset["test"]
)
inference_seconds = time.perf_counter() - inference_started

predictions = np.argmax(
    test_results.predictions,
    axis=-1
)

true_labels = test_results.label_ids

report = classification_report(
    true_labels,
    predictions,
    labels=np.arange(len(LABELS)),
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)
report["mean_latency_seconds"] = inference_seconds / len(test_df)

print("\nFINAL CLASSIFICATION REPORT")

print(
    classification_report(
        true_labels,
        predictions,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        zero_division=0
    )
)

# Save report.
with open(
    EVALUATION_DIR / "classification_report.json",
    "w"
) as file:
    json.dump(
        report,
        file,
        indent=4
    )

# Save confusion matrix.
cm = confusion_matrix(
    true_labels,
    predictions,
    labels=np.arange(len(LABELS))
)

pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
).to_csv(
    EVALUATION_DIR / "confusion_matrix.csv"
)

# Save individual predictions.
test_output = test_df.reset_index(drop=True).copy()

test_output["actual_label"] = [
    ID2LABEL[int(x)]
    for x in true_labels
]

test_output["predicted_label"] = [
    ID2LABEL[int(x)]
    for x in predictions
]

test_output["correct"] = (
    test_output["actual_label"]
    == test_output["predicted_label"]
)

test_output.to_csv(
    EVALUATION_DIR / "test_predictions.csv",
    index=False
)

print("\nTest accuracy:", test_results.metrics.get("test_accuracy"))
print("Test macro F1:", report["macro avg"]["f1-score"])

# Save training configuration.
with open(
    OUTPUT_DIR / "training_config.json",
    "w"
) as file:
    json.dump({
        "base_model": BASE_MODEL,
        "labels": LABELS,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "class_weights": weights.tolist()
    }, file, indent=4)


# ============================================================
# OPTIONAL: PRIVATE HUGGING FACE MODEL UPLOAD
# ============================================================

if PUSH_TO_HUB:

    print("\nPublishing model privately...")

    api = HfApi(token=HF_TOKEN)

    api.create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        private=True,
        exist_ok=True
    )

    trainer.model.push_to_hub(
        MODEL_REPO,
        private=True,
        token=HF_TOKEN
    )

    tokenizer.push_to_hub(
        MODEL_REPO,
        private=True,
        token=HF_TOKEN
    )

    print(
        "Model uploaded:",
        f"https://huggingface.co/{MODEL_REPO}"
    )


print("\nTRAINING PIPELINE COMPLETED")
print("Model:", OUTPUT_DIR)
print("Evaluation:", EVALUATION_DIR)
