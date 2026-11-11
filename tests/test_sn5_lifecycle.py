"""Check bounded cache cleanup and native saved-model validation, without GPU dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("sn5_driver", Path(__file__).resolve().parents[1] / "scripts/sn5_driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class LifecycleTests(unittest.TestCase):
    def test_cleanup_preserves_training_pairs_and_compact_hashes_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            phase = work / "data/iter1"
            phase.mkdir(parents=True)
            (phase / "1_future_judgement.json").write_text(json.dumps([{"scores":[5, 1], "responses":["a", "b"]}]))
            (phase / "1_future_response.json").write_text("[]")
            (phase / "D1.official.json").write_text("[]")
            pairs = phase / "D1/train.json"
            pairs.parent.mkdir()
            pairs.write_text("canonical pairs")
            snapshot = work / "configs/Mf1.data"
            snapshot.mkdir(parents=True)
            (snapshot / "train.json").write_text("canonical pairs")
            driver.discard_consumed(work, "iter1", ["Mf1"])
            manifest = work / "metrics/iter1-consumed-caches.json"
            before = manifest.read_bytes()
            self.assertEqual(len(json.loads(before)), 3)
            self.assertEqual(pairs.read_text(), "canonical pairs")
            self.assertFalse(snapshot.exists())
            self.assertFalse((phase / "1_future_judgement.json").exists())
            driver.discard_consumed(work, "iter1", ["Mf1"])
            self.assertEqual(manifest.read_bytes(), before)

    def test_cleanup_refuses_unexpected_snapshot_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            p = work / "configs/M1.data/keep.txt"
            p.parent.mkdir(parents=True)
            p.write_text("keep")
            with self.assertRaises(RuntimeError):
                driver.discard_consumed(work, "iter0", ["M1"])
            self.assertTrue(p.exists())

    def test_model_must_have_updates_and_every_weight_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            p = work / "models/M1"
            p.mkdir(parents=True)
            (p / ".done").touch()
            (p / "trainer_state.json").write_text(json.dumps({"global_step":2, "log_history":[{"loss":0.6}]}))
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"w":"model.safetensors"}}))
            with self.assertRaises(RuntimeError):
                driver.model_summary(work, "M1")
            (p / "model.safetensors").write_bytes(b"test")
            self.assertEqual(driver.model_summary(work, "M1")["global_step"], 2)
            (p / "trainer_state.json").write_text(json.dumps({"global_step":2, "log_history":[{"loss":float("nan")}]}))
            with self.assertRaises(RuntimeError):
                driver.model_summary(work, "M1")


if __name__ == "__main__":
    unittest.main()
