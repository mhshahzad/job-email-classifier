import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
SOURCE_FILE = ROOT / "email_dataset" / "hf_upload.jsonl"
OUTPUT_DIR = ROOT / "training_dataset"
ANALYSIS_PLOT = OUTPUT_DIR / "dataset_splits_analysis.png"
SYNTHETIC_OFFER_FILES = (
    OUTPUT_DIR / "synthetic_offers.jsonl",
    OUTPUT_DIR / "synthetic_offer_scenarios.jsonl",
)
SYNTHETIC_ASSESSMENT_FILE = OUTPUT_DIR / "synthetic_assessments.jsonl"
SPLIT_FILES = {
    "train": OUTPUT_DIR / "train.jsonl",
    "validation": OUTPUT_DIR / "validation.jsonl",
    "test": OUTPUT_DIR / "test.jsonl",
}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42


def load_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path} on line {line_number}")
            if not {"subject", "body", "label"}.issubset(row):
                raise ValueError(f"{path} line {line_number} is missing required fields")
            if not all(
                isinstance(row[field], str) for field in ("subject", "body", "label")
            ):
                raise ValueError(
                    f"Subject, body, and label must be strings on line {line_number}"
                )
            records.append(
                {"subject": row["subject"], "body": row["body"], "label": row["label"]}
            )
    return records


def text_key(record: dict[str, str]) -> tuple[str, str]:
    return record["subject"].strip().casefold(), record["body"].strip().casefold()


def clean_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, int]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[text_key(record)].append(record)

    conflicting_groups = {
        key
        for key, group in grouped.items()
        if len({record["label"] for record in group}) > 1
    }

    cleaned = []
    seen = set()
    for record in records:
        key = text_key(record)
        if key in conflicting_groups or key in seen:
            continue
        cleaned.append(record)
        seen.add(key)

    removed_count = len(records) - len(cleaned)
    return cleaned, removed_count, len(conflicting_groups)


def split_grouped_records(
    records: list[dict[str, str]], seed: int = RANDOM_SEED
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[text_key(record)].append(record)

    groups_by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in grouped.values():
        dominant_label = Counter(record["label"] for record in group).most_common(1)[0][0]
        groups_by_label[dominant_label].append(group)

    label_totals = Counter(record["label"] for record in records)
    target_label_counts = {
        label: {
            split: total * ratio for split, ratio in SPLIT_RATIOS.items()
        }
        for label, total in label_totals.items()
    }
    target_total_counts = {
        split: len(records) * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    current_label_counts = {
        label: Counter() for label in label_totals
    }
    current_total_counts = Counter()
    splits = {split: [] for split in SPLIT_RATIOS}
    randomizer = random.Random(seed)

    def placement_cost(label: str, split: str, group_size: int) -> float:
        cost = 0.0
        for candidate in SPLIT_RATIOS:
            label_count = current_label_counts[label][candidate]
            total_count = current_total_counts[candidate]
            if candidate == split:
                label_count += group_size
                total_count += group_size
            label_target = target_label_counts[label][candidate]
            total_target = target_total_counts[candidate]
            cost += (label_count - label_target) ** 2 / max(label_target, 1)
            cost += (total_count - total_target) ** 2 / max(total_target, 1)
        return cost

    for label, groups in groups_by_label.items():
        randomizer.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for group in groups:
            group_size = len(group)
            split = min(
                SPLIT_RATIOS,
                key=lambda candidate: placement_cost(label, candidate, group_size),
            )
            splits[split].extend(group)
            current_label_counts[label][split] += group_size
            current_total_counts[split] += group_size

    return splits


def load_synthetic_records(path: Path, expected_label: str) -> list[dict[str, str]]:
    records = load_records(path)
    invalid_labels = sorted(
        {record["label"] for record in records if record["label"] != expected_label}
    )
    if invalid_labels:
        raise ValueError(
            f"Synthetic examples in {path} must all have label "
            f"{expected_label!r}; found {invalid_labels}"
        )
    return records


def load_synthetic_training_records(
    source_records: list[dict[str, str]],
    offer_files: tuple[Path, ...],
    assessment_file: Path,
) -> tuple[list[dict[str, str]], int, int]:
    offers = [
        record
        for path in offer_files
        for record in load_synthetic_records(path, "offer")
    ]
    assessments = load_synthetic_records(assessment_file, "assessment")
    synthetic = offers + assessments

    source_text = {text_key(record) for record in source_records}
    if any(text_key(record) in source_text for record in synthetic):
        raise ValueError("A synthetic example duplicates source text")
    if len({text_key(record) for record in synthetic}) != len(synthetic):
        raise ValueError("Synthetic example files contain duplicate subject/body text")
    return synthetic, len(offers), len(assessments)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def analyze_splits(
    splits: dict[str, list[dict[str, str]]], plot_file: Path
) -> None:
    split_data = pd.concat(
        {
            split_name: pd.DataFrame.from_records(
                records, columns=["subject", "body", "label"]
            )
            for split_name, records in splits.items()
        },
        names=["split", "row"],
    ).reset_index(level="split")
    split_data["subject_characters"] = split_data["subject"].str.len()
    split_data["body_words"] = split_data["body"].str.split().str.len()

    for split_name, split_records in split_data.groupby("split", sort=False):
        label_counts = split_records["label"].value_counts().sort_index()
        label_percentages = (
            split_records["label"].value_counts(normalize=True).sort_index() * 100
        )
        duplicate_count = split_records.duplicated(
            subset=["subject", "body"]
        ).sum()
        empty_subjects = split_records["subject"].str.strip().eq("").sum()
        empty_bodies = split_records["body"].str.strip().eq("").sum()

        print(f"{split_name.title()} records: {len(split_records)}")
        print(f"Unique labels: {len(label_counts)}")
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({label_percentages[label]:.1f}%)")
        print(f"Duplicate subject/body records: {duplicate_count}")
        print(f"Empty subjects: {empty_subjects}")
        print(f"Empty bodies: {empty_bodies}")

    label_counts_by_split = split_data.groupby(["label", "split"]).size().unstack(
        fill_value=0
    )
    median_lengths = split_data.groupby(["label", "split"])[
        ["subject_characters", "body_words"]
    ].median()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(21, 7))
    label_counts_by_split.plot(
        kind="bar", ax=axes[0], title="Examples by category and split"
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
        axis.legend(title="Split")
    axes[0].set_ylabel("Example count")
    axes[1].set_ylabel("Median characters")
    axes[2].set_ylabel("Median words")
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)
    print(f"Split analysis visualization: {plot_file.resolve()}")


def main() -> None:
    source_records = load_records(SOURCE_FILE)
    records, removed_count, conflicting_groups = clean_records(source_records)
    splits = split_grouped_records(records)
    real_split_counts = {split: len(rows) for split, rows in splits.items()}
    synthetic, synthetic_offer_count, synthetic_assessment_count = (
        load_synthetic_training_records(
            source_records,
            SYNTHETIC_OFFER_FILES,
            SYNTHETIC_ASSESSMENT_FILE,
        )
    )
    splits["train"].extend(synthetic)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in SPLIT_FILES.items():
        write_jsonl(path, splits[split])

    print(f"Source records: {len(source_records)}")
    print(f"Records removed during cleanup: {removed_count}")
    print(f"Conflicting-label duplicate groups removed: {conflicting_groups}")
    print(f"Cleaned source records: {len(records)}")
    print(f"Real source split sizes (target 70/15/15): {real_split_counts}")
    print(f"Prebuilt synthetic offers added to training only: {synthetic_offer_count}")
    print(
        "Prebuilt synthetic assessments added to training only: "
        f"{synthetic_assessment_count}"
    )
    for split, split_records in splits.items():
        print(
            f"{split}: {len(split_records)} records; "
            f"labels: {dict(sorted(Counter(r['label'] for r in split_records).items()))}"
        )
    analyze_splits(splits, ANALYSIS_PLOT)
    print("Grouping safeguard: exact normalized subject/body matches stay together.")
    print(
        "Thread IDs are unavailable in the source; this does not guarantee "
        "conversation-level separation."
    )


if __name__ == "__main__":
    main()
