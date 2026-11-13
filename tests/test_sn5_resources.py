"""Validate real resource plans and completed-stage reuse without a GPU allocation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_sn5")
driver = load("sn5_driver")


class ResourcesTests(unittest.TestCase):
    def test_eight_gpu_plan_keeps_batch_and_sn5_constraint(self):
        plan = prepare.resource_plan(8, "any-sn5")
        self.assertEqual(plan["sft_gradient_accumulation"], 2)
        self.assertEqual(plan["dpo_gradient_accumulation"], 4)
        self.assertEqual(plan["requests"]["nvidia.com/gpu"], "8")
        self.assertEqual(plan["limits"]["nvidia.com/gpu"], "8")
        selectors = {x["key"]:x["values"] for x in plan["node_expressions"]}
        self.assertEqual(selectors["group"], ["shennong-5", "shennong-5-dev"])
        self.assertEqual(selectors["node.kubernetes.io/instance-type"], ["ml.p5e.48xlarge", "ml.p5en.48xlarge"])
        self.assertNotIn("kubernetes.io/hostname", selectors)

    def test_default_node_is_preserved_until_explicitly_overridden(self):
        selectors = prepare.resource_plan(8, prepare.NODE)["node_expressions"]
        self.assertEqual(selectors[-1]["values"], [prepare.NODE])
        for n in (2, 4, 8):
            p = prepare.resource_plan(n, prepare.NODE)
            self.assertEqual(n * p["sft_micro_batch"] * p["sft_gradient_accumulation"], 32)
            self.assertEqual(n * p["dpo_micro_batch"] * p["dpo_gradient_accumulation"], 32)
        for n in (0, 3, 16):
            with self.assertRaises(ValueError):
                prepare.resource_plan(n, prepare.NODE)

    def fixture(self, root, *, done=True, epoch=2.98):
        source, dest = root / "source", root / "dest"
        plan = dict(upstream_commit="commit", base_model="base", base_directory="/base", global_batch=32,
                    sft_micro_batch=2, sft_epochs=3, sft_lr=2e-6, warmup_ratio=.1, train_seed=42)
        for run, gpus in [(source, 2), (dest, 8)]:
            driver.atomic(run / "run-plan.json", dict(plan, gpus=gpus))
            for name in ["dataset-manifest.json", "base-model-manifest.json"]:
                driver.atomic(run / name, {"hash":"identical"})
            for rel in ["third_party/TSR/alignment-handbook/scripts/run_sft.py",
                        "third_party/TSR/settings/sft/config/config.yaml"]:
                p = run / "bundle" / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("same source")
        model = source / "trial/models/M0"
        model.mkdir(parents=True)
        if done:
            (model / ".done").touch()
        (model / "model.safetensors").write_bytes(b"weights")
        driver.atomic(model / "trainer_state.json", dict(global_step=180, epoch=epoch, log_history=[{"loss":1.0}]))
        cfg = source / "trial/configs/M0.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("max_steps: -1\nnum_train_epochs: 3\n")
        return source, dest

    def test_completed_stage_is_reused_without_weight_copy_or_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = self.fixture(Path(tmp))
            driver.inherit_completed_sft(dest, source)
            model = dest / "trial/models/M0"
            self.assertTrue(model.is_symlink())
            self.assertEqual(model.resolve(), source / "trial/models/M0")
            provenance = json.loads((dest / "sft-provenance.json").read_text())
            self.assertFalse(provenance["optimizer_resume"])
            self.assertFalse(provenance["training_repeated"])
            self.assertEqual(provenance["source_gpus"], 2)
            self.assertEqual(provenance["files"]["model.safetensors"]["sha256"], driver.sha256(model / "model.safetensors"))
            self.assertTrue((dest / "sft-complete.json").is_file())
            self.assertFalse((source / "trial/metrics").exists())

    def test_incomplete_sft_is_never_reused(self):
        for done, epoch in [(False, 2.98), (True, .5)]:
            with self.subTest(done=done, epoch=epoch), tempfile.TemporaryDirectory() as tmp:
                source, dest = self.fixture(Path(tmp), done=done, epoch=epoch)
                with self.assertRaises(RuntimeError):
                    driver.inherit_completed_sft(dest, source)
                self.assertFalse((dest / "sft-complete.json").exists())
                self.assertFalse((dest / "trial/models/M0").exists())

    def test_changed_dataset_or_code_prevents_reuse(self):
        for change in ["dataset", "code"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                source, dest = self.fixture(Path(tmp))
                if change == "dataset":
                    driver.atomic(dest / "dataset-manifest.json", {"hash":"different"})
                else:
                    (dest / "bundle/third_party/TSR/alignment-handbook/scripts/run_sft.py").write_text("changed")
                with self.assertRaises(RuntimeError):
                    driver.inherit_completed_sft(dest, source)
                self.assertFalse((dest / "sft-complete.json").exists())


if __name__ == "__main__":
    unittest.main()
