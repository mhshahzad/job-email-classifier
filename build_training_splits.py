import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE_FILE = ROOT / "email_dataset" / "hf_upload.jsonl"
OUTPUT_DIR = ROOT / "training_dataset"
SYNTHETIC_FILE = OUTPUT_DIR / "synthetic_offers.jsonl"
SPLIT_FILES = {
    "train": OUTPUT_DIR / "train.jsonl",
    "validation": OUTPUT_DIR / "validation.jsonl",
    "test": OUTPUT_DIR / "test.jsonl",
}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42

SUBJECTS = (
    "Offer of Employment for [ROLE]",
    "Your [ROLE] Offer from [COMPANY]",
    "Formal Offer: [ROLE] Position",
    "Next Steps: [ROLE] Offer",
    "Employment Offer and Details",
)
OPENINGS = (
    "We are pleased to offer you the [ROLE] position at [COMPANY].",
    "Following our conversations, [COMPANY] would like to offer you the [ROLE] role.",
    "We are delighted to extend an offer for you to join us as [ROLE].",
    "This email confirms our offer for the [ROLE] position with [COMPANY].",
    "The team enjoyed meeting you, and we would like to offer you the [ROLE] position.",
    "We would be excited to have you join [COMPANY] as [ROLE]. Please review our offer below.",
    "Your interviews are complete, and we are ready to make you an offer for [ROLE].",
    "On behalf of [COMPANY], I am happy to present our offer for the [ROLE] role.",
    "We are moving forward with an offer for you to join [COMPANY] as [ROLE].",
    "It is my pleasure to share the details of our offer for the [ROLE] position.",
)
DETAILS = (
    "The position is full-time and remote, with a proposed start date to be agreed together.",
    "The role is based in [LOCATION], and the expected start date is flexible by mutual agreement.",
    "The annual base salary is [SALARY], with benefits described in the attached offer letter.",
    "This offer includes [BENEFITS]; the complete terms are listed in the attached document.",
    "The position is full-time, reporting to [MANAGER], with compensation detailed in the offer letter.",
    "We have included the proposed compensation, benefits, and employment terms in the attached letter.",
    "The role is [WORK ARRANGEMENT], and the compensation package is outlined in the attached document.",
    "Your proposed start date is [START_DATE], subject to your confirmation and completion of onboarding.",
    "The offer is contingent on the standard pre-employment checks described in the attached terms.",
    "Please review the attached document for salary, benefits, and other terms of employment.",
)
NEXT_STEPS = (
    "Please review the attached letter and let us know your decision by [RESPONSE_DATE].",
    "If you would like to accept, reply to this message and we will send the agreement for signature.",
    "You can discuss any part of the offer with me before the response date of [RESPONSE_DATE].",
    "Please let us know if you have questions or would like to discuss any of the terms.",
    "To accept, sign and return the offer document by [RESPONSE_DATE].",
)
SIGNOFFS = (
    "Best regards,\n[RECRUITER]\n[COMPANY] Talent Team",
    "Kind regards,\n[HIRING_MANAGER]\n[COMPANY]",
    "Sincerely,\n[RECRUITER]\nPeople Operations",
    "Thank you,\n[HIRING_MANAGER]\n[COMPANY] Recruiting",
    "Warm regards,\n[RECRUITER]\nTalent Acquisition",
)


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


def generate_synthetic_offers(count: int, seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    variants = [
        (subject, opening, details, next_step, signoff)
        for subject in SUBJECTS
        for opening in OPENINGS
        for details in DETAILS
        for next_step in NEXT_STEPS
        for signoff in SIGNOFFS
    ]
    if count > len(variants):
        raise ValueError(
            f"Requested {count} synthetic examples, but only "
            f"{len(variants)} template combinations are available"
        )
    random.Random(seed).shuffle(variants)
    records = []
    for subject, opening, details, next_step, signoff in variants[:count]:
        records.append(
            {
                "subject": subject,
                "body": (
                    f"Hello [CANDIDATE],\n\n{opening} {details} "
                    f"{next_step}\n\n{signoff}"
                ),
                "label": "offer",
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic 70/15/15 label-stratified splits, keeping "
            "identical normalized subject/body text together."
        )
    )
    parser.add_argument(
        "--synthetic-offers",
        type=int,
        default=50,
        help="Synthetic offer examples to add to training only (default: 50).",
    )
    args = parser.parse_args()
    if args.synthetic_offers < 0:
        parser.error("--synthetic-offers must be non-negative")

    records = load_records(SOURCE_FILE)
    splits = split_grouped_records(records)
    real_split_counts = {split: len(rows) for split, rows in splits.items()}
    synthetic = generate_synthetic_offers(args.synthetic_offers)
    existing_text = {text_key(record) for record in records}
    if any(text_key(record) in existing_text for record in synthetic):
        raise ValueError("A synthetic example duplicates source email text")
    splits["train"].extend(synthetic)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(SYNTHETIC_FILE, synthetic)
    for split, path in SPLIT_FILES.items():
        write_jsonl(path, splits[split])

    print(f"Source emails: {len(records)}")
    print(f"Real source split sizes (target 70/15/15): {real_split_counts}")
    print(f"Synthetic offers added to training only: {len(synthetic)}")
    for split, split_records in splits.items():
        print(
            f"{split}: {len(split_records)} records; "
            f"labels: {dict(sorted(Counter(r['label'] for r in split_records).items()))}"
        )
    print("Grouping safeguard: exact normalized subject/body matches stay together.")
    print(
        "Thread IDs are unavailable in the source; this does not guarantee "
        "conversation-level separation."
    )


if __name__ == "__main__":
    main()
