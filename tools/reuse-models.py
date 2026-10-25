#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Complete an update by reusing hash-verified model files from a base release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--from',dest='base',required=True,type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
base=a.base.resolve()
if base==root:raise SystemExit('Base release must be a different directory')
manifest=json.loads((root/'manifest.json').read_text())
if manifest.get('reuse_from')!='0.1.0':raise SystemExit('Not a supported update archive')
for name in manifest['reused_files']:
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise SystemExit('Invalid model path')
    src=base/relative;dst=root/relative;expected=manifest['files'][name]
    with src.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if src.stat().st_size!=expected['bytes'] or sha!=expected['sha256']:
        raise SystemExit('Base model mismatch: '+name)
    if dst.exists():
        with dst.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected['sha256']:raise SystemExit('Destination already contains a different file: '+name)
        continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(src,dst)
    except OSError:shutil.copyfile(src,dst)
subprocess.run([sys.executable,str(root/'tools/verify-install.py'),str(root)],check=True)
print('Update ready:',root)
