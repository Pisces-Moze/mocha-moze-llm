#!/usr/bin/env python3
"""Verify all files in an extracted release against its content manifest."""
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
for name, expected in manifest['files'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise SystemExit('Invalid manifest path')
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise SystemExit('Missing or incorrect file: ' + name)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected['sha256']:
        raise SystemExit('SHA256 mismatch: ' + name)
print('Verified', len(manifest['files']), 'release files')
