"""Exercise the real datasets loader and production config/pair helpers, without GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from datasets import load_dataset
import yaml

REPO = Path(__file__).resolve().parents[1]


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tsr data ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        for mode, rel, micro, accumulation, epochs in [
            ("dpo", "sr/config/dpo/config_full_iter0.yaml", 2, 2, 1),
            ("sft", "sft/config/config.yaml", 4, 1, 3),
        ]:
            path = self.upstream / "settings" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(dict(per_device_train_batch_size=micro,
                            gradient_accumulation_steps=accumulation, num_train_epochs=epochs, seed=42)))
        self.env = dict(os.environ, TSR_DIR=str(self.upstream), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        self.data = self.root / "pairs"
        self.records = [{"prompt": f"question {i}", "chosen": "good", "rejected": "bad"} for i in range(3)]
        write_rows(self.data / "train.json", self.records)
        self.cfg = self.root / "configs/M2.yaml"

    def config(self, mode="dpo", check=True, ngpu=2):
        return subprocess.run([sys.executable, str(REPO / "src/make_cfg.py"), mode,
            "--name", "M2", "--init", "base-model", "--data", str(self.data),
            "--out_dir", str(self.root / "model"), "--ngpu", str(ngpu), "--cfg", str(self.cfg)],
            env=self.env, text=True, capture_output=True, check=check)

    def configured_rows(self):
        config = yaml.safe_load(self.cfg.read_text())
        path = next(iter(config["dataset_mixer"]))
        dataset = load_dataset(path, split="train", cache_dir=str(self.root / "cache"))
        return config, Path(path), list(dataset)

    def test_legacy_duplicate_is_read_once_and_preserved(self):
        duplicate = self.data / "official/train.json"
        write_rows(duplicate, self.records)
        old = load_dataset(str(self.data), split="train", cache_dir=str(self.root / "legacy-cache"))
        self.assertEqual(len(old), 6)  # Reproduce the original regression with the actual loader.
        self.config()
        config, path, loaded = self.configured_rows()
        self.assertEqual(loaded, self.records)
        self.assertEqual(list(path.iterdir()), [path / "train.json"])
        self.assertEqual(duplicate.read_bytes(), (self.data / "train.json").read_bytes())
        self.assertEqual(config["num_train_epochs"], 1)
        self.assertEqual(config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * 2, 32)

    def test_unrelated_training_files_are_not_imported(self):
        write_rows(self.data / "nested/train_extra.json", [{"prompt": "must not train", "chosen": "x", "rejected": "y"}])
        self.config()
        self.assertEqual(self.configured_rows()[2], self.records)

    def test_regenerated_config_refreshes_the_snapshot(self):
        self.config()
        updated = self.records + [{"prompt": "new question", "chosen": "new good", "rejected": "new bad"}]
        write_rows(self.data / "train.json", updated)
        self.config()
        self.assertEqual(self.configured_rows()[2], updated)

    def test_contaminated_snapshot_fails_before_training(self):
        self.config()
        extra = self.cfg.with_suffix(".data") / "official/train.json"
        write_rows(extra, self.records)
        result = self.config(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected files", result.stderr)
        self.assertTrue(extra.exists())

    def test_sft_still_loads_multiple_training_files(self):
        (self.data / "train.json").unlink()
        write_rows(self.data / "ift_train.jsonl", self.records[:2])
        write_rows(self.data / "eft_train.jsonl", self.records[2:])
        self.config(mode="sft")
        config, path, loaded = self.configured_rows()
        self.assertEqual(path, self.data)
        self.assertCountEqual(loaded, self.records)
        self.assertEqual(config["num_train_epochs"], 3)

    def test_eight_gpus_preserve_batch_and_scientific_settings(self):
        for mode, micro, accumulation in [("sft", 2, 2), ("dpo", 1, 4)]:
            with self.subTest(mode=mode):
                self.config(mode=mode, ngpu=2)
                previous = yaml.safe_load(self.cfg.read_text())
                self.config(mode=mode, ngpu=8)
                current = yaml.safe_load(self.cfg.read_text())
                self.assertEqual(current["per_device_train_batch_size"], micro)
                self.assertEqual(current["gradient_accumulation_steps"], accumulation)
                self.assertEqual(8 * micro * accumulation, 32)
                previous.pop("gradient_accumulation_steps")
                current.pop("gradient_accumulation_steps")
                self.assertEqual(current, previous)

    def test_comparison_files_stay_outside_new_pair_directories(self):
        chosen = self.root / "judgements.json"
        chosen.write_text(json.dumps([dict(prompt="question", responses=["best", "worst"], scores=[5, 1])]))
        # Stub only the external upstream formatter; test the production pair helper's output placement.
        formatter = """import argparse, os, shutil
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', required=True)
args, _ = parser.parse_known_args()
shutil.copyfile(os.environ['PAIR_REFERENCE'], args.output_path)
"""
        scripts = self.upstream / "settings/tsr"
        scripts.mkdir(parents=True)
        for mode in ["past", "future"]:
            with self.subTest(mode=mode):
                (scripts / f"reformat2DPO_{mode}.py").write_text(formatter)
                reference = self.root / (mode + "-reference")
                args = ["--chosen", str(chosen), "--rejected", str(chosen)]
                if mode == "future":
                    args += ["--future", str(chosen)]
                subprocess.run([sys.executable, str(REPO / "src/build_pairs.py"), mode, *args,
                                "--out", str(reference)], check=True, capture_output=True)
                out = self.root / mode / "D2"
                env = dict(self.env, PAIR_REFERENCE=str(reference / "train.json"), TEST_REPO=str(REPO),
                           TEST_PY=sys.executable, TEST_LOGS=str(self.root / "logs"))
                script = '. "$TEST_REPO/scripts/lib.sh"; PY="$TEST_PY"; LOGS="$TEST_LOGS"; mkdir -p "$LOGS"; pairs "$@"'
                inputs = [str(chosen), str(chosen)] + ([str(chosen)] if mode == "future" else [])
                subprocess.run(["bash", "-euc", script, "test-pairs", mode, str(out) + "/", *inputs],
                               env=env, check=True, capture_output=True)
                self.assertTrue(out.with_suffix(".official.json").exists())
                self.assertFalse((out / "official").exists())
                self.assertTrue(json.loads((out / "stats.json").read_text())["matches_official"])
                dataset = load_dataset(str(out), split="train", cache_dir=str(self.root / "cache"))
                self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
