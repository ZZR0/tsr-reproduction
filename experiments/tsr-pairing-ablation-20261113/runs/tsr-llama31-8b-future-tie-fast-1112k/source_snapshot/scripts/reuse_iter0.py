"""Copy only completed, verified M0 generation/judge data into a fresh run."""
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def inherit_iter0_data(run, source):
    run, source = run.resolve(), source.resolve()
    assert source != run and source.parent == run.parent
    old = json.loads((source / "run-plan.json").read_text())
    new = json.loads((run / "run-plan.json").read_text())
    for key in ["base_model", "upstream_commit", "pair_selection", "train_seed", "gpus"]:
        assert old[key] == new[key], key
    for name in ["dataset-manifest.json", "base-model-manifest.json"]:
        assert json.loads((source/name).read_text()) == json.loads((run/name).read_text())
    for name in ["source_model", "files"]:
        assert json.loads((source/"sft-provenance.json").read_text())[name] == json.loads((run/"sft-provenance.json").read_text())[name]
    for key in ["K", "temperature", "top_p", "max_tokens", "judge_count", "data_parallel_shards"]:
        assert old["generation"][key] == new["generation"][key]
    assert sha(source/"bundle/src/build_pairs.py") == sha(run/"bundle/src/build_pairs.py")
    c = json.loads((source/"trial/data/iter0/0_chosen_response.json").read_text())
    j = json.loads((source/"trial/data/iter0/0_chosen_judgement.json").read_text())
    assert len(c) == len(j) == 5000
    assert all(x["prompt"] == y["prompt"] and x["responses"] == y["responses"]
               and len(x["responses"]) == len(y["scores"]) == 7 for x,y in zip(c,j))
    stats = json.loads((source/"trial/data/iter0/D1/stats.json").read_text())
    assert stats["strict_score_gap"] and stats["retained_tie_pairs"] == 0 and stats["gap_min"] > 0
    assert stats["rejected_tie_break"] == "current" and stats["chosen_policy"] == "current_best"
    names=["0_chosen_response.json", "0_chosen_judgement.json", "0_chosen_judgement.stats.json", "D1/train.json", "D1/stats.json"]
    evidence=[]
    for rel in names:
        src, dst = source/"trial/data/iter0"/rel, run/"trial/data/iter0"/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        assert not dst.exists()
        shutil.copy2(src,dst)
        digest=sha(src)
        assert digest == sha(dst)
        evidence.append(dict(path=rel,bytes=src.stat().st_size,sha256=digest))
    (run/"iter0-data-provenance.json").write_text(json.dumps(dict(source_run=str(source),
        reused_prompts=5000,reused_candidates=35000,reused_judgments=35000,
        dpo_weights_reused=False,files=evidence),indent=2)+"\n")
