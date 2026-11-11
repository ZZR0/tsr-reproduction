"""Exercise actual pair selection and the default iter1 entry, without model dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def candidate(prompt, prefix, scores, responses=None):
    return dict(prompt=prompt, responses=responses or [f"{prefix}-{i}" for i in range(len(scores))], scores=scores)


class FutureOnlyPairsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tsr future only ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def inputs(self, current, past, future):
        paths = []
        for name, values in zip(["chosen", "rejected", "future"], [current, past, future]):
            p = self.root / f"1_{name}_judgement.json"
            p.write_text(json.dumps(values))
            paths.append(p)
        return paths

    def build(self, current, past, future, *flags):
        c, r, f = self.inputs(current, past, future)
        out = self.root / "D2"
        result = subprocess.run([sys.executable, str(REPO / "src/build_pairs.py"), "future",
            "--chosen", str(c), "--rejected", str(r), "--future", str(f), "--out", str(out),
            *flags], text=True, capture_output=True, check=True)
        rows = [json.loads(line) for line in (out / "train.json").read_text().splitlines()]
        return rows, json.loads((out / "stats.json").read_text())

    def test_future_wins_even_when_current_is_better_and_gap_filter_is_implicit(self):
        current = [candidate(p, "current", s) for p, s in [("keep", [5, 1]), ("tie", [5, 3]), ("reverse", [5, 4])]]
        past = [candidate(p, "past", [4, 2]) for p in ["keep", "tie", "reverse"]]
        past[1]["scores"] = [4, 3]
        past[2]["scores"] = [5, 4]
        future = [candidate(p, "future", s) for p, s in [("keep", [3, 2]), ("tie", [3, 3]), ("reverse", [2, 1])]]
        rows, stats = self.build(current, past, future, "--future-only")
        self.assertEqual([r["prompt"] for r in rows], ["keep"])
        self.assertEqual(rows[0]["chosen"][-1]["content"], "future-0")
        self.assertEqual(rows[0]["rejected"][-1]["content"], "current-1")
        self.assertEqual(stats["filtered_out"], 2)
        self.assertEqual(stats["retained_chosen_from_Mf"], 1)
        self.assertEqual(stats.get("retained_chosen_from_Mi", 0), 0)
        self.assertEqual(stats["retained_tie_pairs"], 0)
        self.assertEqual(stats["gap_min"], 2)

    def test_rejected_pool_includes_past_and_ties_prefer_current(self):
        current = [candidate("a", "current", [5, 2]), candidate("b", "current", [5, 1])]
        past = [candidate("a", "past", [4, 1]), candidate("b", "past", [4, 1])]
        future = [candidate("a", "future", [5, 0]), candidate("b", "future", [5, 0])]
        rows, _ = self.build(current, past, future, "--future-only")
        self.assertEqual([r["rejected"][-1]["content"] for r in rows], ["past-1", "current-1"])

    def test_length_tiebreak_never_imports_current_answers(self):
        for rule, expected in [("longest", "future-long"), ("shortest", "f")]:
            with self.subTest(rule=rule):
                rows, _ = self.build([candidate("p", "c", [5, 5, 1], ["x" * 50, "c", "bad"])],
                    [candidate("p", "r", [2])], [candidate("p", "f", [5, 5], ["f", "future-long"])],
                    "--future-only", "--cho_len", rule)
                self.assertEqual(rows[0]["chosen"][-1]["content"], expected)

    def test_original_selection_remains_available_explicitly(self):
        rows, stats = self.build([candidate("p", "current", [5, 1])],
            [candidate("p", "past", [2])], [candidate("p", "future", [3])])
        self.assertEqual(rows[0]["chosen"][-1]["content"], "current-0")
        self.assertEqual(stats["chosen_policy"], "current_future_best")

    def test_empty_result_is_reported_without_retaining_ties(self):
        rows, stats = self.build([candidate("p", "c", [5])], [candidate("p", "r", [5])],
            [candidate("p", "f", [5])], "--future-only")
        self.assertEqual(rows, [])
        self.assertIsNone(stats["gap_min"])
        self.assertEqual(stats["n_pairs"], 0)

    def test_incompatible_options_fail_before_input_loading(self):
        for options in [["past", "--future-only"], ["future", "--future-only", "--strict"],
                        ["future", "--future-only", "--score", "conf"]]:
            with self.subTest(options=options):
                p = subprocess.run([sys.executable, str(REPO / "src/build_pairs.py"), *options,
                    "--chosen", "missing", "--rejected", "missing", "--future", "missing", "--out", "missing"],
                    capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertNotIn("Traceback", p.stderr)

    def pipeline(self, future_scores):
        data = self.root / "data/iter1"
        data.mkdir(parents=True)
        for p in self.inputs([candidate("p", "current", [5, 1])],
                             [candidate("p", "past", [4, 2])], [candidate("p", "future", future_scores)]):
            (data / p.name).write_bytes(p.read_bytes())
        # A stale pair file must be rebuilt, not accepted simply because it exists.
        stale = self.root / "data/iter1_future_only/D2"
        stale.mkdir(parents=True)
        (stale / "train.json").write_text('{"prompt":"stale"}\n')
        for name in ["M0", "M1"]:
            p = self.root / "models" / name
            p.mkdir(parents=True)
            (p / ".done").touch()
        script = r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"
LOGS="$TEST_ROOT/logs"; PY="$TEST_PY"; TSR_DIR="$TEST_ROOT/upstream"
mkdir -p "$LOGS"
need() { for f in "$@"; do [ -e "$f" ] || return 1; done; }
gen() { :; }; judge() { :; }; pairs() { :; }
dpo() { echo "$1|$2|$3" >> "$TEST_ROOT/dpo-calls"; }
stage_iter1
'''
        env = dict(os.environ, TEST_REPO=str(REPO), TEST_ROOT=str(self.root), TEST_PY=sys.executable)
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True), stale

    def test_default_pipeline_rebuilds_d2_and_uses_separate_model(self):
        result, out = self.pipeline([3, 2])
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in (out / "train.json").read_text().splitlines()]
        self.assertEqual(rows[0]["chosen"][-1]["content"], "future-0")
        calls = (self.root / "dpo-calls").read_text().splitlines()
        self.assertTrue(calls[-1].startswith("M2_future_only|"), calls)
        self.assertTrue(calls[-1].endswith("iter1_future_only/D2"), calls)

    def test_empty_filtered_data_stops_before_m2_training(self):
        result, out = self.pipeline([1, 0])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((out / "train.json").read_text(), "")
        self.assertNotIn("M2", (self.root / "dpo-calls").read_text())


if __name__ == "__main__":
    unittest.main()
