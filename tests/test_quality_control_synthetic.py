from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from quality_control_synthetic import (
    build_report,
    combined_text,
    main,
    ollama_judge_batch,
    run_deterministic_checks,
)


class QualityControlSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "source_file": "synthetic_offers.jsonl",
                "line_number": 1,
                "subject": "Offer details",
                "body": "We offer you the role.",
                "label": "offer",
            },
            {
                "source_file": "synthetic_assessments.jsonl",
                "line_number": 1,
                "subject": "Offer details",
                "body": "We offer you the role.",
                "label": "assessment",
            },
        ]

    def test_combined_text_matches_training_input_format(self):
        self.assertEqual(
            combined_text(self.records[0]),
            "Subject: Offer details\nEmail: We offer you the role.",
        )
        record = dict(self.records[0], subject=" <b>Offer</b>  details ")
        self.assertEqual(
            combined_text(record),
            "Subject: Offer details\nEmail: We offer you the role.",
        )

    def test_exact_duplicate_with_different_labels_is_flagged(self):
        checks = run_deterministic_checks(self.records)

        self.assertIn("exact_label_overlap", checks[0]["deterministic_issues"])
        self.assertIn("exact_label_overlap", checks[1]["deterministic_issues"])

    def test_ollama_batch_uses_structured_json_assessments(self):
        response_body = (
            b'{"message":{"content":"{'
            b'\\"assessments\\":[{\\"record_index\\":0,'
            b'\\"quality_score\\":4,\\"label_match\\":true,'
            b'\\"label_ambiguity\\":false,\\"issues\\":[],'
            b'\\"reason\\":\\"Clear offer.\\"}]}"}}'
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return response_body

        with patch(
            "quality_control_synthetic.urlopen", return_value=FakeResponse()
        ) as mocked:
            assessments = ollama_judge_batch(
                self.records[:1],
                start_index=0,
                model="test-model",
                timeout=1,
            )

        self.assertEqual(assessments[0]["record_index"], 0)
        self.assertTrue(assessments[0]["label_match"])
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertIsNone(request.get_header("Authorization"))
        request_payload = json.loads(request.data)
        self.assertEqual(request_payload["model"], "test-model")
        self.assertFalse(request_payload["stream"])
        self.assertEqual(request_payload["format"]["type"], "object")

    def test_ollama_http_error_includes_model_and_response(self):
        error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            404,
            "Not Found",
            hdrs=None,
            fp=BytesIO(b"model not found"),
        )

        with patch("quality_control_synthetic.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Ollama.*404.*test-model"):
                ollama_judge_batch(
                    self.records[:1],
                    start_index=0,
                    model="test-model",
                    timeout=1,
                )

    def test_ollama_connection_error_suggests_starting_server(self):
        with patch(
            "quality_control_synthetic.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Start Ollama"):
                ollama_judge_batch(
                    self.records[:1],
                    start_index=0,
                    model="test-model",
                    timeout=1,
                )

    def test_local_model_progress_advances_after_successful_batch(self):
        progress = MagicMock()
        progress_context = MagicMock()
        progress_context.__enter__.return_value = progress
        assessments = [
            {
                "record_index": index,
                "quality_score": 4,
                "label_match": True,
                "label_ambiguity": False,
                "issues": [],
                "reason": "Clear label.",
            }
            for index in range(len(self.records))
        ]
        with (
            patch(
                "quality_control_synthetic.tqdm",
                return_value=progress_context,
            ) as tqdm_mock,
            patch(
                "quality_control_synthetic.ollama_judge_batch",
                return_value=assessments,
            ),
        ):
            from quality_control_synthetic import run_ollama_judge

            result = run_ollama_judge(self.records, "test-model")

        tqdm_mock.assert_called_once_with(
            total=len(self.records),
            desc="Local model quality review",
            unit="record",
        )
        progress.update.assert_called_once_with(len(self.records))
        self.assertEqual(len(result), len(self.records))

    def test_cli_uses_model_option_and_ollama_model_environment(self):
        test_cases = (
            (["quality_control_synthetic.py", "--model", "llama3.2:latest"], {}),
            (
                ["quality_control_synthetic.py"],
                {"OLLAMA_MODEL": "llama3.2:latest"},
            ),
        )
        for arguments, environment in test_cases:
            with self.subTest(arguments=arguments, environment=environment):
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "report.json"
                    arguments = [
                        *arguments,
                        "--output",
                        str(output_path),
                    ]
                    with (
                        patch.object(sys, "argv", arguments),
                        patch.dict(os.environ, environment, clear=True),
                        patch(
                            "quality_control_synthetic.read_records",
                            return_value=self.records,
                        ),
                        patch(
                            "quality_control_synthetic.run_ollama_judge",
                            return_value=[],
                        ) as judge,
                    ):
                        self.assertEqual(main(), 0)

                    judge.assert_called_once_with(
                        self.records,
                        "llama3.2:latest",
                        300,
                    )

    def test_report_counts_overlap_from_deterministic_and_llm_checks(self):
        checks = run_deterministic_checks(self.records)
        llm_assessments = [
            {
                "record_index": index,
                "quality_score": 4,
                "label_match": False,
                "label_ambiguity": True,
                "issues": ["ambiguous intent"],
                "reason": "Could fit both classes.",
            }
            for index in range(len(self.records))
        ]

        report = build_report(self.records, checks, llm_assessments, "test-model")

        self.assertEqual(report["records_reviewed"], 2)
        self.assertEqual(report["judge_provider"], "Ollama")
        self.assertEqual(report["judge_model"], "test-model")
        self.assertEqual(report["duplicate_records"], 0)
        self.assertEqual(report["label_overlap_records"], 2)


if __name__ == "__main__":
    unittest.main()
