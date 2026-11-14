import unittest

from email_inference import format_classification, format_email_text


class EmailInferenceTests(unittest.TestCase):
    def test_format_email_text_matches_model_input(self):
        self.assertEqual(
            format_email_text("Interview", "Choose a time."),
            "Subject: Interview\nEmail: Choose a time.",
        )

    def test_format_classification_returns_rounded_scores_and_label(self):
        result = format_classification(
            [0.123456, 0.876544],
            {0: "application_confirmation", 1: "interview_invitation"},
        )

        self.assertEqual(
            result,
            {
                "predicted_category": "interview_invitation",
                "confidence": 0.8765,
                "all_scores": {
                    "application_confirmation": 0.1235,
                    "interview_invitation": 0.8765,
                },
            },
        )

    def test_format_classification_keeps_first_label_on_tie(self):
        result = format_classification([0.5, 0.5], {0: "offer", 1: "rejection"})

        self.assertEqual(result["predicted_category"], "offer")
        self.assertEqual(result["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
