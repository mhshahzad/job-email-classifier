from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

from build_training_splits import (
    analyze_splits,
    clean_records,
    load_records,
    load_synthetic_records,
    load_synthetic_training_records,
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

    def test_clean_records_removes_normalized_duplicates_and_conflicts(self):
        records = [
            {"subject": "Interview", "body": "You are invited.", "label": "interview"},
            {"subject": " INTERVIEW ", "body": "you are invited. ", "label": "interview"},
            {"subject": "Offer", "body": "Please review.", "label": "offer"},
            {"subject": " offer ", "body": "please review. ", "label": "rejection"},
            {"subject": "Assessment", "body": "Complete the test.", "label": "assessment"},
        ]

        cleaned, removed_count, conflicting_groups = clean_records(records)

        self.assertEqual(
            cleaned,
            [
                {"subject": "Interview", "body": "You are invited.", "label": "interview"},
                {"subject": "Assessment", "body": "Complete the test.", "label": "assessment"},
            ],
        )
        self.assertEqual(removed_count, 3)
        self.assertEqual(conflicting_groups, 1)

    def test_analysis_reports_splits_and_writes_aggregate_plot(self):
        splits = {
            "train": [
                {"subject": "Offer", "body": "Review your offer.", "label": "offer"},
                {"subject": "Thanks", "body": "Application received.", "label": "confirmation"},
            ],
            "validation": [
                {"subject": "Assessment", "body": "Complete the test.", "label": "assessment"},
            ],
            "test": [
                {"subject": "Interview", "body": "Choose a time.", "label": "interview"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            plot_file = Path(directory) / "analysis.png"
            output = StringIO()
            with redirect_stdout(output):
                analyze_splits(splits, plot_file)

            self.assertTrue(plot_file.is_file())
            self.assertIn("Train records: 2", output.getvalue())
            self.assertIn("Label distribution:", output.getvalue())
            self.assertIn("Duplicate subject/body records: 0", output.getvalue())

    def test_load_synthetic_records_checks_expected_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            path.write_text(
                '{"subject":"Offer","body":"Please review the offer.","label":"offer"}\n',
                encoding="utf-8",
            )

            records = load_synthetic_records(path, "offer")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], "offer")

    def test_load_synthetic_records_rejects_unexpected_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            path.write_text(
                '{"subject":"Offer","body":"Please review the offer.","label":"assessment"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must all have label"):
                load_synthetic_records(path, "offer")

    def test_prebuilt_offer_files_and_assessments_are_combined_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_offers = root / "synthetic_offers.jsonl"
            scenario_offers = root / "synthetic_offer_scenarios.jsonl"
            assessments = root / "synthetic_assessments.jsonl"
            original_offers.write_text(
                '{"subject":"Offer A","body":"Original offer.","label":"offer"}\n',
                encoding="utf-8",
            )
            scenario_offers.write_text(
                '{"subject":"Offer B","body":"Scenario offer.","label":"offer"}\n',
                encoding="utf-8",
            )
            assessments.write_text(
                '{"subject":"Assessment","body":"Complete the task.","label":"assessment"}\n',
                encoding="utf-8",
            )

            synthetic, offer_count, assessment_count = load_synthetic_training_records(
                [],
                (original_offers, scenario_offers),
                assessments,
            )

            self.assertEqual(offer_count, 2)
            self.assertEqual(assessment_count, 1)
            self.assertEqual([record["label"] for record in synthetic], ["offer", "offer", "assessment"])

    def test_load_records_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"subject":"Hello","body":"World"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
