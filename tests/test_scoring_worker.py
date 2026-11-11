import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("scoring_worker", Path(__file__).resolve().parents[1] / "scripts/score_when_ready.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ScoringTests(unittest.TestCase):
    def test_retry_keeps_cached_annotations_and_stops_after_three_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "annotations.json"
            cache.write_text('[{"preference":1.2}]')
            def failing_call(script, arguments, env, log):
                log.write_text("openai.RateLimitError: Error code: 429 - expired_secret_key")
                raise RuntimeError("API failed")
            with patch.object(worker, "call", side_effect=failing_call) as mock, patch.object(worker.time, "sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    worker.score_with_retries(None, [], {}, root, "M0", {})
                self.assertEqual(mock.call_count, 3)
                self.assertEqual(sleep.call_count, 2)
            self.assertEqual(cache.read_text(), '[{"preference":1.2}]')
            self.assertEqual(len(list(root.glob('M0-*-attempt*.log'))), 3)

    def test_auth_or_data_errors_are_not_retried(self):
        for text in ["Error code: 401 - invalid_api_key", "Error code: 403", "ValueError: missing logprobs"]:
            with self.subTest(text=text):
                self.assertFalse(worker.retryable_log(text))

    def test_partial_writer_and_wrong_order_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "outputs.json"
            refs = [{"instruction":str(i)} for i in range(805)]
            self.assertFalse(worker.input_ready(p, refs))
            p.write_text('[{"instruction":')
            self.assertFalse(worker.input_ready(p, refs))
            p.write_text(json.dumps([dict(x, output="") for x in refs[:804]]))
            self.assertFalse(worker.input_ready(p, refs))
            rows = [dict(x, output="") for x in refs]
            p.write_text(json.dumps(rows))
            self.assertTrue(worker.input_ready(p, refs))  # Empty model answers remain part of evaluation.
            p.write_text(json.dumps(rows[::-1]))
            with self.assertRaises(RuntimeError):
                worker.input_ready(p, refs)

    def test_only_full_finite_weighted_scores_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            def leaderboard(n=805):
                with (p / "leaderboard.csv").open("w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["", "n_total", "win_rate", "length_controlled_winrate", "avg_length"])
                    writer.writerow(["M0", n, 12.1, 10.4, 1300])
            leaderboard()
            annotations = [{"preference":1.2} for _ in range(805)]
            (p / "annotations.json").write_text(json.dumps(annotations))
            self.assertEqual(worker.validate_result(p, "M0")["n_total"], 805)
            leaderboard(804)
            with self.assertRaises(RuntimeError):
                worker.validate_result(p, "M0")
            leaderboard()
            annotations[0]["preference"] = float("nan")
            (p / "annotations.json").write_text(json.dumps(annotations))
            with self.assertRaises(RuntimeError):
                worker.validate_result(p, "M0")


if __name__ == "__main__":
    unittest.main()
