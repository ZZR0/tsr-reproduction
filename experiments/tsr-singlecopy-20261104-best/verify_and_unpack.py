#!/usr/bin/env python3
"""Verify every archived byte; optionally restore a portable data directory."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Unsafe manifest path: ' + relative)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, help='Restore decompressed canonical files into an empty directory.')
    ap.add_argument('--restore-duplicates', action='store_true', help='Also recreate byte-identical original input/verification copies.')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit('Output directory must be empty.')
        args.output.mkdir(parents=True, exist_ok=True)
    verified = {}
    restored = 0
    for entry in manifest['files']:
        stored = entry['stored_path']
        if stored in verified:
            assert verified[stored] == (entry['bytes'], entry['sha256']), entry['path']
            if args.output and args.restore_duplicates:
                canonical = safe_path(args.output, entry['identical_to'])
                target = safe_path(args.output, entry['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(canonical.read_bytes())
                restored += 1
            continue
        encoded = safe_path(root, stored).read_bytes()
        assert len(encoded) == entry['stored_bytes'], stored
        assert hashlib.sha256(encoded).hexdigest() == entry['stored_sha256'], stored
        payload = gzip.decompress(encoded) if entry['compression'] == 'gzip' else encoded
        assert len(payload) == entry['bytes'], entry['path']
        assert hashlib.sha256(payload).hexdigest() == entry['sha256'], entry['path']
        verified[stored] = (len(payload), entry['sha256'])
        if args.output:
            target = safe_path(args.output, entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            restored += 1
    print(json.dumps({'status': 'PASS', 'source_files': len(manifest['files']),
                      'unique_payloads': len(verified), 'restored_files': restored}, indent=2))


if __name__ == '__main__':
    main()
