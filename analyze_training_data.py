import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DATASET_DIR = Path(__file__).parent / "training_dataset"
DATASET_SPLITS = {
    "training": DATASET_DIR / "train.jsonl",
    "validation": DATASET_DIR / "validation.jsonl",
    "test": DATASET_DIR / "test.jsonl",
}


def load_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} on line {line_number}"
                )
            if not {"subject", "body", "label"}.issubset(row):
                raise ValueError(
                    f"{path} line {line_number} is missing required fields"
                )
            if not all(
                isinstance(row[field], str)
                for field in ("subject", "body", "label")
            ):
                raise ValueError(
                    f"Subject, body, and label must be strings on line {line_number}"
                )
            records.append(row)
    return records


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} on line {line_number}"
                )
            if not {"label", "predicted_label"}.issubset(row):
                raise ValueError(
                    f"{path} line {line_number} must contain label and predicted_label"
                )
            if not isinstance(row["label"], str) or not isinstance(
                row["predicted_label"], str
            ):
                raise ValueError(
                    f"Labels must be strings in {path} on line {line_number}"
                )
            predictions.append(
                {"label": row["label"], "predicted_label": row["predicted_label"]}
            )
    if not predictions:
        raise ValueError(f"No predictions found in {path}")
    return pd.DataFrame.from_records(
        predictions, columns=["label", "predicted_label"]
    )


def report_classification_metrics(predictions: pd.DataFrame) -> None:
    labels = sorted(set(predictions["label"]) | set(predictions["predicted_label"]))
    confusion = pd.crosstab(
        pd.Categorical(predictions["label"], categories=labels),
        pd.Categorical(predictions["predicted_label"], categories=labels),
        dropna=False,
    )
    confusion.index = labels
    confusion.columns = labels
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = pd.Series(
        [confusion.loc[label, label] for label in labels], index=labels
    )
    precision = true_positive.div(predicted_count.replace(0, pd.NA)).fillna(0)
    recall = true_positive.div(support.replace(0, pd.NA)).fillna(0)
    f1 = (2 * precision * recall).div((precision + recall).replace(0, pd.NA)).fillna(0)

    print("Test classification metrics:")
    print("  label | support | precision | recall | F1")
    for label in labels:
        print(
            f"  {label} | {support[label]} | {precision[label]:.3f} | "
            f"{recall[label]:.3f} | {f1[label]:.3f}"
        )
    accuracy = true_positive.sum() / len(predictions) if len(predictions) else 0
    weights = support / support.sum() if support.sum() else support
    weighted_f1 = (f1 * weights).sum()
    print(
        f"  overall | support={len(predictions)} | accuracy={accuracy:.3f} | "
        f"macro-F1={f1.mean():.3f} | weighted-F1={weighted_f1:.3f}"
    )


def analyze(
    datasets: dict[str, list[dict[str, str]]],
    plot_file: Path,
    predictions: pd.DataFrame | None = None,
) -> None:
    emails = pd.concat(
        {
            dataset_name: pd.DataFrame.from_records(
                records, columns=["subject", "body", "label"]
            )
            for dataset_name, records in datasets.items()
        },
        names=["dataset", "row"],
    ).reset_index(level="dataset")
    emails["subject_characters"] = emails["subject"].str.len()
    emails["body_words"] = emails["body"].str.split().str.len()

    for dataset_name, split_emails in emails.groupby("dataset", sort=False):
        label_counts = split_emails["label"].value_counts().sort_index()
        label_percentages = (
            split_emails["label"].value_counts(normalize=True).sort_index() * 100
        )
        duplicate_emails = split_emails.duplicated(
            subset=["subject", "body"]
        ).sum()
        empty_subjects = split_emails["subject"].str.strip().eq("").sum()
        empty_bodies = split_emails["body"].str.strip().eq("").sum()

        print(f"{dataset_name.title()} records: {len(split_emails)}")
        print(f"Unique labels: {len(label_counts)}")
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({label_percentages[label]:.1f}%)")
        print(f"Duplicate subject/body records: {duplicate_emails}")
        print(f"Empty subjects: {empty_subjects}")
        print(f"Empty bodies: {empty_bodies}")

    if predictions is not None:
        report_classification_metrics(predictions)
    else:
        print(
            "Classification metrics not calculated: provide model predictions "
            "with --predictions to score the test split."
        )

    label_counts_by_dataset = emails.groupby(["label", "dataset"]).size().unstack(
        fill_value=0
    )
    median_lengths = emails.groupby(["label", "dataset"])[
        ["subject_characters", "body_words"]
    ].median()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(21, 7))
    label_counts_by_dataset.plot(
        kind="bar", ax=axes[0], title="Emails by category and dataset"
    )
    median_lengths["subject_characters"].unstack().plot(
        kind="bar", ax=axes[1], title="Median subject length (characters)"
    )
    median_lengths["body_words"].unstack().plot(
        kind="bar", ax=axes[2], title="Median body length (words)"
    )
    for axis in axes:
        axis.set_xlabel("Label")
        axis.tick_params(axis="x", labelrotation=25)
        axis.legend(title="Dataset")
    axes[0].set_ylabel("Email count")
    axes[1].set_ylabel("Median characters")
    axes[2].set_ylabel("Median words")
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)
    print(f"Combined visualization: {plot_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize the dataset splits."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Analyze a single JSONL file instead of all train/validation/test splits.",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        help="Output PNG path when analyzing a single file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "JSONL with test ground-truth `label` and model `predicted_label` "
            "fields for per-class precision, recall, and F1."
        ),
    )
    args = parser.parse_args()
    if args.path:
        plot_file = args.plot_file or args.path.with_name(
            f"{args.path.stem}_analysis.png"
        )
        analyze(
            {args.path.stem: load_records(args.path)},
            plot_file,
            load_predictions(args.predictions) if args.predictions else None,
        )
        return

    if args.plot_file:
        parser.error("--plot-file can only be used when analyzing one file")

    datasets = {
        dataset_name: load_records(path)
        for dataset_name, path in DATASET_SPLITS.items()
    }
    predictions = load_predictions(args.predictions) if args.predictions else None
    if predictions is not None:
        test_labels = [record["label"] for record in datasets["test"]]
        if predictions["label"].tolist() != test_labels:
            raise ValueError(
                "Prediction rows must match the test split labels in the same order"
            )
    analyze(
        datasets,
        DATASET_DIR / "dataset_splits_analysis.png",
        predictions,
    )


if __name__ == "__main__":
    main()
