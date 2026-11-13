"""Disambiguate current/future source ties from chosen/rejected preference ties."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def candidate(prompt, prefix, scores):
    return dict(prompt=prompt, responses=[f"{prefix}-{i}" for i in range(len(scores))], scores=scores)


class CurrentFutureTieTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tsr current future ties ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, TEST_REPO=str(REPO), TEST_ROOT=str(self.root), TEST_PY=sys.executable)

    def test_sources_choose_higher_and_current_on_ties_then_filter_equal_preferences(self):
        prompts = ["source-tie", "future-higher", "current-higher", "pair-tie"]
        current = [candidate(p, "current", s) for p,s in zip(prompts, [[5,1],[4,1],[5,1],[5,5]])]
        future = [candidate(p, "future", s) for p,s in zip(prompts, [[5,2],[5,2],[4,2],[5,5]])]
        past = [candidate(p, "past", s) for p,s in zip(prompts, [[5,1],[5,1],[5,1],[5,5]])]
        for name,rows in [("current",current),("future",future),("past",past)]:
            (self.root / f"{name}.json").write_text(json.dumps(rows))
        for length in ["first", "longest", "shortest"]:
            with self.subTest(length=length):
                out = self.root / length
                subprocess.run([sys.executable,str(REPO/"src/build_pairs.py"),"future",
                    "--chosen",str(self.root/"current.json"),"--rejected",str(self.root/"past.json"),
                    "--future",str(self.root/"future.json"),"--out",str(out),"--strict","--filter",
                    "--rej_tie","mi","--cho_len",length],check=True,capture_output=True,text=True)
                rows=[json.loads(x) for x in (out/"train.json").read_text().splitlines()]
                self.assertEqual([x["prompt"] for x in rows],prompts[:3])
                self.assertEqual([x["chosen"][-1]["content"] for x in rows],
                                 ["current-0","future-0","current-0"])
                self.assertEqual([x["rejected"][-1]["content"] for x in rows],["current-1"]*3)
                stats=json.loads((out/"stats.json").read_text())
                self.assertEqual(stats["chosen_tie_break"],"current")
                self.assertEqual(stats["filtered_out"],1)
                self.assertEqual(stats["retained_tie_pairs"],0)
                self.assertEqual(stats["retained_chosen_from_Mi"],2)
                self.assertEqual(stats["retained_chosen_from_Mf"],1)

    def test_default_two_round_pipeline_passes_current_ties_and_filtering_to_real_builder(self):
        # Actual shell stage -> pair helper -> production builder, with only GPU operations stubbed.
        for phase,files in {
            "iter0":{"0_chosen_judgement.json":candidate("p","current0",[5,1]),
                     "0_future_judgement.json":candidate("p","future0",[5,2])},
            "iter1":{"1_chosen_judgement.json":candidate("p","current1",[5,1]),
                     "1_rejected_judgement.json":candidate("p","past1",[5,1]),
                     "1_future_judgement.json":candidate("p","future1",[5,2])},
        }.items():
            folder=self.root/"data"/phase;folder.mkdir(parents=True)
            for name,row in files.items():(folder/name).write_text(json.dumps([row]))
        script=r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
need() { :; }; gen() { :; }; judge() { :; }
dpo() { printf "%s\n" "$1" >> "$TEST_ROOT/model-calls"; }
stage_iter0
stage_iter1
'''
        result=subprocess.run(["bash","-c",script],env=self.env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.root/"model-calls").read_text().splitlines(),["Mf0","M1","Mf1","M2_current_tie"])
        for phase in ["iter0","iter1_current_tie"]:
            d=self.root/"data"/phase/"D2"
            pair=json.loads((d/"train.json").read_text())
            self.assertEqual(pair["chosen"][-1]["content"],"current0-0" if phase=="iter0" else "current1-0")
            stats=json.loads((d/"stats.json").read_text())
            self.assertTrue(stats["strict_score_gap"])
            self.assertEqual(stats["chosen_tie_break"],"current")
        for phase in ["iter0","iter1"]:
            stats=json.loads((self.root/"data"/phase/"D1/stats.json").read_text())
            self.assertTrue(stats["strict_score_gap"])

    def test_cached_future_preferred_ties_are_rejected_even_with_positive_gap(self):
        out=self.root/"D2";out.mkdir()
        (out/"train.json").write_text('{"prompt":"p"}\n')
        (out/"stats.json").write_text(json.dumps(dict(n_pairs=1,strict_score_gap=True,
             gap_min=4,retained_tie_pairs=0,rejected_tie_break="current",chosen_tie_break="future")))
        script=r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
pairs future "$TEST_ROOT/D2" missing missing missing -- --strict --filter --rej_tie mi
'''
        result=subprocess.run(["bash","-c",script],env=self.env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("chosen current/future tie-breaking rule differs",result.stderr)
        self.assertIn("fresh WORK_DIR",result.stderr)


if __name__ == "__main__":
    unittest.main()
