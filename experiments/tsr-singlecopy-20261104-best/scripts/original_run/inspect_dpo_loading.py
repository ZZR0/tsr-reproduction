"""Check the real datasets loader's row multiplicity before comparing DPO runs."""
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset, load_dataset_builder

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--repetitions', type=int, required=True)
args = parser.parse_args()
canonical = (args.directory / 'train.json').read_bytes()
pair_count = len(canonical.splitlines())
assert pair_count > 0
builder = load_dataset_builder(str(args.directory))
files = list(builder.config.data_files['train'])
assert len(files) == args.repetitions, (files, args.repetitions)
for path in files:
    assert Path(path).read_bytes() == canonical, 'Unexpected or nonidentical training file: ' + path
dataset = load_dataset(str(args.directory), split='train')
assert len(dataset) == pair_count * args.repetitions
print(json.dumps(dict(directory=str(args.directory), source_pairs=pair_count,
                      repetitions=args.repetitions, loaded_rows=len(dataset),
                      expected_steps=len(dataset) // 32, files=files,
                      canonical_sha256=hashlib.sha256(canonical).hexdigest())))
