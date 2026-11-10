# Job email classifier

An NLP project for classifying job-search emails into application
confirmation, assessment, interview invitation, offer, and rejection. It
compares fine-tuned DistilBERT and BERT-base models, TF-IDF baselines, and
locally served Ollama models.

**[Open the interactive evaluation dashboard](portfolio/dashboard.html)**

The dashboard is a standalone HTML file and uses aggregate metrics only.

![Evaluation dashboard](portfolio/blog_artifacts/dashboard.png)

## Results

All models were evaluated on the same fixed held-out split of 973 emails.
Trainable models selected checkpoints using validation macro-F1.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Fine-tuned BERT-base | 0.963 | 0.8586 |
| Fine-tuned DistilBERT | 0.962 | 0.844 |
| TF-IDF + Linear SVM | 0.959 | 0.799 |
| TF-IDF + Logistic Regression | 0.947 | 0.807 |
| TF-IDF + Complement Naive Bayes | 0.900 | 0.508 |

![Held-out accuracy comparison](portfolio/blog_artifacts/model_accuracy.png)

![Held-out macro-F1 comparison](portfolio/blog_artifacts/model_macro_f1.png)

These are single-run results, not a multi-seed benchmark. The test split has
only two offer examples, so offer metrics are highly uncertain. The labels
are project reference labels rather than an independently adjudicated gold
set. Thread IDs were unavailable, so conversation-level separation between
splits cannot be confirmed. The BERT-base checkpoint emitted missing and
unexpected LayerNorm-key warnings; validate those weights before relying on
that result.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, evaluation context, and
limitations.

## Engineering highlights

- Fine-tuning and evaluation pipelines for DistilBERT and BERT-base, with
  checkpoint selection by validation macro-F1.
- Classical TF-IDF baselines and optional local Ollama comparisons using
  aggregate metrics.
- Deterministic, label-stratified data splitting that keeps identical
  normalized subject/body pairs together.
- Synthetic offer examples added only to training data to address a sparse
  class.
- Unit tests for split generation and synthetic data; GitHub Actions runs
  them on pushes and pull requests.
- Privacy-conscious evaluation artifacts: no email text or row-level
  predictions are embedded in the public dashboard.

## Privacy

The public showcase contains aggregate results and visualizations only. It
does not include message text, row-level predictions, credentials, or model
checkpoints.

## Project code

- `train.py` — DistilBERT fine-tuning and held-out evaluation
- `compare_models.py` — classical baselines, BERT-base, and optional Ollama evaluation
- `predict.py` — local inference with the saved DistilBERT checkpoint
- `portfolio/dashboard.html` — standalone HTML evaluation dashboard
