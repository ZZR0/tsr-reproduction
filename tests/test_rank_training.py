"""Real two-process datasets loading/shuffling with isolated caches and identical samples."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


class RankCacheTests(unittest.TestCase):
    def test_tensorboard_local_write_and_shared_snapshot_preserve_scalars(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        spec = importlib.util.spec_from_file_location("rank_training", REPO / "src/rank_training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, TSR_TENSORBOARD_LOCAL_ROOT=str(root / "local")):
                local = module.tensorboard_directory(root / "cfg.yaml")
                writer = SummaryWriter(str(local))
                writer.add_scalar("loss", 0.5, 1)
                writer.add_scalar("loss", 0.25, 2)
                writer.close()
                module.sync_tensorboard(root / "cfg.yaml", root / "shared")
                events = EventAccumulator(str(root / "shared" / local.name)).Reload()
                self.assertEqual([(e.step,e.value) for e in events.Scalars("loss")], [(1,0.5),(2,0.25)])

    def test_two_ranks_keep_same_shuffle_and_separate_arrow_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.json"
            data.write_text("".join(json.dumps({"id":i}) + "\n" for i in range(100)))
            native = root / "native.py"
            native.write_text('''import json, sys
from pathlib import Path
from datasets import load_dataset, config
d = load_dataset("json", data_files=sys.argv[1], split="train").shuffle(seed=42)
Path(sys.argv[2]).write_text(json.dumps({"ids":d["id"], "cache":str(config.HF_DATASETS_CACHE), "files":[x["filename"] for x in d.cache_files]}))
''')
            def execute(rank):
                out = root / f"rank{rank}.json"
                env = dict(os.environ, LOCAL_RANK=str(rank), TSR_RANK_CACHE_ROOT=str(root / "cache"),
                           HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
                subprocess.run([sys.executable,str(REPO / "src/rank_training.py"),str(native),str(data),str(out)],
                               env=env,check=True,capture_output=True,text=True)
                return json.loads(out.read_text())
            with ThreadPoolExecutor(max_workers=2) as pool:
                a,b = list(pool.map(execute,[0,1]))
            self.assertEqual(a["ids"],b["ids"])
            self.assertNotEqual(a["ids"],list(range(100)))
            self.assertNotEqual(a["cache"],b["cache"])
            self.assertTrue(set(a["files"]).isdisjoint(b["files"]))
            for row in [a,b]:
                self.assertTrue(all(Path(f).is_relative_to(Path(row["cache"])) for f in row["files"]))


if __name__ == "__main__":
    unittest.main()
