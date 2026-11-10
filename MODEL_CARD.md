# Model card: Job email classifier

## Overview

This project fine-tunes DistilBERT and BERT-base to classify job-search
emails into five categories: `application_confirmation`, `assessment`,
`interview_invitation`, `offer`, and `rejection`. TF-IDF classifiers and
locally served language models are included as comparison systems.

## Intended use

The models are for experimentation and educational evaluation of
classification methods. They may help organize an individual's own
job-search correspondence, with human review of every prediction.

They are not validated for hiring, candidate screening, employment
decisions, or other consequential use. Do not use a prediction as the sole
basis for an employment-related decision.

## Evaluation

Models use a fixed held-out test split of 973 examples. Checkpoints for
trainable models are selected using validation macro-F1. Reported
fine-tuned results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| BERT-base | 0.963 | 0.8586 |
| DistilBERT | 0.962 | 0.844 |

These are single-run results, not a multi-seed benchmark. The test set
contains only two offer examples, making that class's metrics highly
uncertain. Labels are not independently adjudicated. Conversation-level
separation cannot be confirmed because thread identifiers were unavailable.
The BERT-base run emitted missing/unexpected LayerNorm-key warnings during
checkpoint loading; its weights should be verified before relying on its
reported score.

## Limitations and risks

- Class imbalance can lead to high overall accuracy while minority-class
  performance remains uncertain.
- Message templates, language, and recruiting practices may differ from
  the model's training examples and cause performance to shift.
- Predictions can be wrong or overconfident; verify the email itself.
- Latency depends on hardware, batching, and runtime and is not a
  deployment guarantee.

## Privacy and deployment

Email content can contain sensitive personal information. The public
showcase contains aggregate results only; message text, row-level
predictions, credentials, model checkpoints, and experiment notebooks are
not included.

No model weights are distributed in this repository.
