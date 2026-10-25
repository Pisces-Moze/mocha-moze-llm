#!/usr/bin/env python3
"""Create a self-contained release from tested ARMhf binaries and pinned models.

Arguments point to private build artifacts. No vendor compiler/toolkit is copied.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-engine', required=True, type=Path)
    p.add_argument('--stories-engine', required=True, type=Path)
    p.add_argument('--qwen-weights', required=True, type=Path)
    p.add_argument('--stories-inputs', required=True, type=Path)
    p.add_argument('--runtime-prefix', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='new build directory')
    p.add_argument('--version', default='0.2.0', choices=['0.1.0','0.2.0'])
    p.add_argument('--reuse-models', action='store_true', help='update archive; reuse verified weights from v0.1.0')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = a.output / ('mocha-llm-v'+a.version)
    root.mkdir(parents=True, exist_ok=False)
    for name in ['src', 'tools', 'models', 'licenses', 'docs', 'tests', 'runtime-source', 'ci']:
        shutil.copytree(repo / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['mocha-llm', 'README.md', 'NOTICE', 'LICENSE']:
        shutil.copyfile(repo / name, root / name)
    # Archives built from a Windows checkout must contain runnable POSIX scripts.
    for source in root.rglob('*'):
        if source.is_file():
            raw=source.read_bytes()
            source.write_bytes(raw.replace(b'\r\n',b'\n'))
    (root / 'bin').mkdir()
    for source, name in [(a.qwen_engine, 'mocha-qwen-engine'), (a.stories_engine, 'mocha-stories-engine')]:
        raw = source.read_bytes()
        if raw[:5] != b'\x7fELF\x01' or int.from_bytes(raw[18:20], 'little') != 40:
            raise ValueError('Not an ELF32 ARM executable: ' + str(source))
        shutil.copyfile(source, root / 'bin' / name)
        (root / 'bin' / name).chmod(0o755)
    reusable={'models/qwen-q8.bin':a.qwen_weights,
              'models/stories260k.bin':a.stories_inputs/'model.bin',
              'models/stories-tokenizer.bin':a.stories_inputs/'tokenizer.bin'}
    if not a.reuse_models:
        for name, source in reusable.items():shutil.copyfile(source,root/name)
    locks = json.loads((repo / 'models/qwen-inputs.json').read_text())
    if digest(a.qwen_weights) != locks['quantized_sha256']:
        raise ValueError('Qwen weight hash mismatch')
    if digest(root / 'models/qwen-tokenizer.json') != locks['tokenizer_sha256']:
        raise ValueError('Qwen tokenizer hash mismatch')
    stories = json.loads((repo / 'models/stories-inputs.json').read_text())['inputs']
    for name, source in [('stories260k.bin', 'model.bin'), ('stories-tokenizer.bin', 'tokenizer.bin')]:
        if digest(a.stories_inputs/source) != stories[source]['sha256']:
            raise ValueError('Stories input mismatch')
    # Only the open runtime and five explicitly enumerated dependency libraries.
    lib = root / 'runtime/lib'
    lib.mkdir(parents=True)
    names = ['libucuda.so.1.0.0', 'libgdev.so.1.0.0', 'libboost_thread.so.1.83.0',
             'libdrm.so.2', 'libdrm_nouveau.so.2']
    for name in names:
        shutil.copyfile(a.runtime_prefix / 'lib' / name, lib / name)
    # Copies instead of symlinks make archives produced on Windows portable.
    for alias, target in {'libucuda.so.1': 'libucuda.so.1.0.0', 'libgdev.so.1': 'libgdev.so.1.0.0',
                          'libcuda.so.1': 'libucuda.so.1.0.0', 'libcudart.so.3': 'libucuda.so.1.0.0'}.items():
        shutil.copyfile(lib / target, lib / alias)
    (root / 'mocha-llm').chmod(0o755)
    files = {str(f.relative_to(root)).replace('\\', '/'): {'bytes': f.stat().st_size, 'sha256': digest(f)}
             for f in sorted(root.rglob('*')) if f.is_file()}
    if a.reuse_models:
        for name, source in reusable.items():files[name]={'bytes':source.stat().st_size,'sha256':digest(source)}
    try:
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        source_commit=None
    manifest = {'version': a.version, 'architecture': 'arm-linux-gnueabihf', 'gpu': 'GK20A sm_32',
                'source_repository':'https://github.com/Pisces-Moze/mocha-moze-llm',
                'source_commit':source_commit,'files': files}
    if a.reuse_models:manifest['reuse_from']='0.1.0';manifest['reused_files']=list(reusable)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = a.output / ('mocha-llm-v'+a.version+'-armhf'+('-update' if a.reuse_models else '')+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(root, arcname=root.name, filter=lambda i: normalize(i))
    (a.output / 'SHA256SUMS').write_text(digest(archive) + '  ' + archive.name + '\n')
    print(archive, archive.stat().st_size, digest(archive))

def normalize(info):
    info.uid = info.gid = 0
    info.uname = info.gname = 'root'
    info.mode = 0o755 if info.isdir() or info.name.endswith(('mocha-llm', 'mocha-qwen-engine', 'mocha-stories-engine')) else 0o644
    return info

if __name__ == '__main__':
    main()
