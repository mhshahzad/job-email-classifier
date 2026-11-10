import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TextIO


def load_records(path: Path) -> list[dict[str, object]]:
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
                    f"{path} line {line_number} is missing subject, body, or label"
                )
            if not all(
                isinstance(row[field], str) for field in ("subject", "body", "label")
            ):
                raise ValueError(
                    f"Subject, body, and label must be strings on line {line_number}"
                )
            records.append(row)
    return records


def text_key(record: dict[str, object]) -> tuple[str, str]:
    subject = record["subject"]
    body = record["body"]
    if not isinstance(subject, str) or not isinstance(body, str):
        raise TypeError("Subject and body must be strings")
    return subject.strip().casefold(), body.strip().casefold()


def clean_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int, int]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[text_key(record)].append(record)

    conflicting_groups = {
        key
        for key, group in groups.items()
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


def write_records(path: Path, records: list[dict[str, object]], *, atomic: bool) -> None:
    if not atomic:
        with path.open("w", encoding="utf-8") as output:
            _write_jsonl(output, records)
        return

    original_mode = path.stat().st_mode & 0o777
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            _write_jsonl(output, records)
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_jsonl(output: TextIO, records: list[dict[str, object]]) -> None:
    for record in records:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate JSONL email rows by normalized subject and body. "
            "Conflicting-label duplicate groups are removed entirely."
        )
    )
    parser.add_argument("path", type=Path, help="JSONL dataset to clean.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to this path instead of replacing the input dataset.",
    )
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.path.resolve():
        parser.error("--output must differ from the input path")

    records = load_records(args.path)
    cleaned, removed_count, conflicting_groups = clean_records(records)
    output_path = args.output or args.path
    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(output_path, cleaned, atomic=not args.output)

    print(f"Input rows: {len(records)}")
    print(f"Rows removed: {removed_count}")
    print(f"Conflicting-label groups removed: {conflicting_groups}")
    print(f"Rows kept: {len(cleaned)}")
    print(f"Cleaned dataset: {output_path.resolve()}")


if __name__ == "__main__":
    main()
