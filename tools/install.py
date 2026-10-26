#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install one complete, pinned v0.4.0 directory; no historical upgrade steps.

The v0.1.0 archive is only a verified model source. Its executable is never
selected. Optional GGUF weights remain opt-in via fetch-models.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = '0.4.0'
INPUTS = [
    ('0.1.0', 'mocha-llm-v0.1.0-armhf.tar.gz', '82c9feddcaca861751e14452d08237880ba72261e3d8dc1c9e99020b505c03a9'),
    (VERSION, 'mocha-llm-v0.4.0-armhf-update.tar.gz', 'af27885f7c2554d7405401d6a9af6d122970c7a5fedd58872a43e34b4eb197c0'),
]


def unpack(archive, target, prefix):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for item in members:
            name = Path(item.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != prefix:
                raise ValueError('Invalid archive path')
            if not (item.isfile() or item.isdir()):
                raise ValueError('Archive links/devices are not accepted')
        tar.extractall(target, members=members, filter='data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path.home()/'Applications/mocha-llm')
    p.add_argument('--reuse-from', type=Path, help='existing verified v0.1.0-v0.4.0 installation; avoids base download')
    p.add_argument('--cache', type=Path, help='directory containing the official archives, for offline installation')
    a = p.parse_args()
    destination = a.directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final = destination/('mocha-llm-v'+VERSION)
    if final.exists():
        subprocess.run([sys.executable, str(final/'tools/verify-install.py'), str(final)], check=True)
        print('Already installed:', final)
        return
    if shutil.disk_usage(destination).free < (128 if a.reuse_from else 1300)*1024**2:
        p.error('Not enough free space; select a data partition or --reuse-from an existing installation')
    with tempfile.TemporaryDirectory(prefix='.install-', dir=destination) as temporary:
        work = Path(temporary)
        base = a.reuse_from.expanduser().resolve() if a.reuse_from else None
        for version, filename, sha in INPUTS:
            if version == '0.1.0' and base:
                continue
            archive = (a.cache/filename) if a.cache else work/filename
            if not a.cache:
                url = f'https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v{version}/{filename}'
                print('Downloading pinned input:', filename, flush=True)
                with urllib.request.urlopen(url, timeout=120) as response, archive.open('xb') as output:
                    shutil.copyfileobj(response, output, length=1024*1024)
            with archive.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                    raise ValueError('Archive SHA256 mismatch: '+filename)
            unpack(archive, work, 'mocha-llm-v'+version)
            if version == '0.1.0':
                base = work/'mocha-llm-v0.1.0'
        staged = work/('mocha-llm-v'+VERSION)
        subprocess.run([sys.executable, str(staged/'tools/reuse-models.py'), '--from', str(base)], check=True)
        (staged/'INSTALLATION.txt').write_text(
            'Mocha LLM v0.4.0 — complete installation\n'
            'This directory contains the current program, runtime and two base models.\n'
            'The v0.1.0 archive was only a model source; no sequential upgrade is needed.\n'
            'Qwen 0.5B defaults to CUDA; 1B/1.5B default to NEON CPU.\n'
            'Q4 CUDA is experimental, currently slower, with unresolved stability limits.\n'
            'Models: ./mocha-llm --list-models\n'
            'Current guide: https://github.com/Pisces-Moze/mocha-moze-llm/blob/main/docs/install-current.md\n'
            'Version table: https://github.com/Pisces-Moze/mocha-moze-llm/blob/main/docs/versions.md\n',encoding='utf-8')
        # Full installation in one directory; hardlinked weights survive removal
        # of the temporary base. verify-install.py checks the released manifest.
        staged.rename(final)
    link = destination/'current'
    if link.exists() and not link.is_symlink():
        raise RuntimeError('Refusing to replace a real current directory')
    temporary_link = destination/'.current-new'
    if temporary_link.exists() or temporary_link.is_symlink():
        raise RuntimeError('Incomplete previous link update; inspect .current-new')
    temporary_link.symlink_to(final.name, target_is_directory=True)
    os.replace(temporary_link, link)
    print('Installed version:', VERSION)
    print('Run:', link/'mocha-llm', '--list-models')


if __name__ == '__main__':
    main()
