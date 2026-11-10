import tempfile
import unittest
from pathlib import Path

from build_training_splits import (
    generate_synthetic_offers,
    load_records,
    split_grouped_records,
)


class BuildTrainingSplitsTests(unittest.TestCase):
    def test_identical_normalized_text_stays_in_one_split(self):
        records = [
            {"subject": "Interview", "body": "You are invited.", "label": "interview_invitation"},
            {"subject": " INTERVIEW ", "body": "you are invited. ", "label": "interview_invitation"},
            {"subject": "Application received", "body": "Thanks for applying.", "label": "application_confirmation"},
            {"subject": "Assessment", "body": "Complete this exercise.", "label": "assessment"},
        ]

        splits = split_grouped_records(records)
        duplicate_split = [
            name
            for name, split_records in splits.items()
            if any(record["subject"] == "Interview" for record in split_records)
        ]

        self.assertEqual(len(duplicate_split), 1)
        grouped_records = [
            record
            for record in splits[duplicate_split[0]]
            if record["body"].strip().casefold() == "you are invited."
        ]
        self.assertEqual(len(grouped_records), 2)

    def test_split_is_deterministic_for_same_seed(self):
        records = [
            {"subject": f"Subject {index}", "body": f"Body {index}", "label": "rejection"}
            for index in range(20)
        ]

        self.assertEqual(
            split_grouped_records(records, seed=7),
            split_grouped_records(records, seed=7),
        )

    def test_synthetic_offers_are_deterministic_and_labeled(self):
        first = generate_synthetic_offers(5, seed=19)
        second = generate_synthetic_offers(5, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(record["label"] == "offer" for record in first))
        self.assertEqual(
            len({(record["subject"], record["body"]) for record in first}),
            5,
        )

    def test_load_records_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"subject":"Hello","body":"World"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
