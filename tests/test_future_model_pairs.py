"""Future-model D1 filtering, source selection and stale-cache safety without GPUs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def row(prompt, prefix, scores):
    return {"prompt": prompt, "responses": [f"{prefix}-{i}" for i in range(len(scores))], "scores": scores}


class FutureModelPairsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tsr filtered future ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.current = self.root / "current.json"
        self.past = self.root / "past.json"
        self.out = self.root / "D1"
        self.env = dict(os.environ, TEST_REPO=str(REPO), TEST_ROOT=str(self.root), TEST_PY=sys.executable)

    def build(self):
        script = r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
pairs past "$TEST_ROOT/D1" "$TEST_ROOT/current.json" "$TEST_ROOT/past.json" -- --filter --rej_tie mi
'''
        return subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True)

    def test_d1_filters_ties_and_preserves_current_chosen_and_rejected_ties(self):
        self.current.write_text(json.dumps([row("tie", "current", [3, 3]),
            row("keep", "current", [4, 1]), row("past-lower", "current", [4, 2])]))
        self.past.write_text(json.dumps([row("tie", "past", [3, 3]),
            row("keep", "past", [5, 1]), row("past-lower", "past", [5, 1])]))
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(x) for x in (self.out / "train.json").read_text().splitlines()]
        self.assertEqual([x["prompt"] for x in rows], ["keep", "past-lower"])
        self.assertEqual([x["chosen"][-1]["content"] for x in rows], ["current-0", "current-0"])
        self.assertEqual([x["rejected"][-1]["content"] for x in rows], ["current-1", "past-1"])
        stats = json.loads((self.out / "stats.json").read_text())
        self.assertTrue(stats["strict_score_gap"])
        self.assertEqual(stats["retained_tie_pairs"], 0)
        self.assertEqual(stats["filtered_out"], 1)
        self.assertEqual(stats["gap_min"], 3)
        self.assertEqual(stats["rejected_tie_break"], "current")
        # Cached filtered data resumes even after consumed judgement files have been removed.
        self.current.unlink(); self.past.unlink()
        self.assertEqual(self.build().returncode, 0)

    def test_equal_best_scores_choose_current_not_past(self):
        self.current.write_text(json.dumps([row("p", "current", [5, 1])]))
        self.past.write_text(json.dumps([row("p", "past", [5, 1])]))
        self.assertEqual(self.build().returncode, 0)
        pair = json.loads((self.out / "train.json").read_text())
        self.assertEqual(pair["chosen"][-1]["content"], "current-0")
        self.assertEqual(pair["rejected"][-1]["content"], "current-1")

    def test_cached_unfiltered_or_different_tiebreak_data_is_rejected_without_overwrite(self):
        self.out.mkdir()
        original = "{\"prompt\":\"historical\"}\n"
        (self.out / "train.json").write_text(original)
        for stats in [dict(strict_score_gap=False, retained_tie_pairs=1, n_pairs=1, gap_min=0),
                      dict(strict_score_gap=True, retained_tie_pairs=0, n_pairs=1, gap_min=1,
                           rejected_tie_break="past")]:
            (self.out / "stats.json").write_text(json.dumps(stats))
            result = self.build()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fresh WORK_DIR", result.stderr)
            self.assertEqual((self.out / "train.json").read_text(), original)

    def test_all_ties_stop_before_publishing_a_training_dataset(self):
        for p in [self.current, self.past]:
            p.write_text(json.dumps([row("p", "same", [3, 3])]))
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.out / "train.json").exists())
        stats = json.loads((self.out.with_suffix(".tmp") / "stats.json").read_text())
        self.assertEqual(stats["n_pairs"], 0)

    def test_both_future_training_stages_pass_filter_and_current_tiebreak(self):
        script = r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
pairs() { printf "%q " "$@" >> "$TEST_ROOT/pair-calls"; printf "\n" >> "$TEST_ROOT/pair-calls"; }
stage_iter0
stage_iter1 main
'''
        result = subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [shlex.split(x) for x in (self.root / "pair-calls").read_text().splitlines()]
        d1 = [x for x in calls if x[1].endswith("/D1")]
        self.assertEqual(len(d1), 2)
        for args in d1:
            self.assertIn("--filter", args)
            self.assertEqual(args[args.index("--rej_tie") + 1], "mi")
        # First-round chosen also compares current/future with current winning ties.
        d2 = next(x for x in calls if x[1].endswith("iter0/D2"))
        self.assertIn("--strict", d2)
        self.assertIn("--filter", d2)


if __name__ == "__main__":
    unittest.main()
