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

The latest reported DistilBERT experiment (v3) used the email subject and
body together, with a maximum sequence length of 256, learning rate of
`2e-5`, seed 42, and eight configured epochs. Its checkpoint was selected
using validation macro-F1. The v3 evaluation covers 973 examples:

| Metric | DistilBERT v3 |
|---|---:|
| Accuracy | 0.9681 |
| Macro-F1 | 0.8673 |
| Weighted-F1 | 0.9677 |
| Validation macro-F1 | 0.8580 |
| Mean latency | 1.90 ms/email |

Per-class v3 test results:

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Application confirmation | 0.9726 | 0.9844 | 0.9785 | 577 |
| Assessment | 0.9091 | 0.7692 | 0.8333 | 13 |
| Interview invitation | 0.9318 | 0.8542 | 0.8913 | 48 |
| Offer | 1.0000 | 0.5000 | 0.6667 | 2 |
| Rejection | 0.9670 | 0.9670 | 0.9670 | 333 |

Earlier reported experiments provide context:

| Experiment | Input | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| DistilBERT v1 | Body only | 0.9620 | 0.8444 | 0.9613 |
| DistilBERT v2-2, expanded synthetic training data | Subject + body | 0.9496 | 0.7094 | 0.9489 |
| DistilBERT v3, latest | Subject + body | 0.9681 | 0.8673 | 0.9677 |
| BERT-base, original benchmark | — | 0.9630 | 0.8586 | 0.9600 |

V2-2 added synthetic offer and assessment examples to training. V3 used the
earlier training sample with subject and body together. The v2-2 and v3
reports reference different dataset repositories/revisions. Their results
are experiment snapshots, not a controlled, paired test of synthetic-data
augmentation or input representation. The earlier BERT-base run emitted
missing/unexpected LayerNorm-key warnings during checkpoint loading, so its
reported score remains provisional. See the [evaluation dashboard](portfolio/dashboard.html)
for aggregate comparisons and the v3 confusion matrix.

These are single-run results, not a multi-seed benchmark. The test set
contains only two offer examples and 13 assessment examples, making those
class metrics highly uncertain. Labels are not independently adjudicated.
Conversation-level separation cannot be confirmed because thread identifiers
were unavailable.

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
