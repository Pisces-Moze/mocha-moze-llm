#!/usr/bin/env python3
"""Fetch hash-pinned build inputs and extract them without system installation."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

def fetch(path, url, sha):
    if not path.exists():
        request=urllib.request.Request(url,headers={'User-Agent':'Mocha-LLM-build'})
        tmp=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(request,timeout=90) as src,tmp.open('wb') as dst:
            while block:=src.read(1024*1024):dst.write(block)
        tmp.replace(path)
    with path.open('rb') as src:
        if hashlib.file_digest(src,'sha256').hexdigest()!=sha:
            raise ValueError('SHA256 mismatch: '+str(path))
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    lock=json.loads((Path(__file__).resolve().parents[1]/'runtime-source/sources.json').read_text())
    downloads=a.directory/'downloads';downloads.mkdir(exist_ok=True)
    inputs=[('cuda-core-6-5_6.5-14_amd64.deb',lock['nvidia_compiler'],'toolkit')]
    inputs += [(n,v,'toolkit') for n,v in lock['nvidia_armhf_inputs'].items()]
    for name,entry,target in inputs:
        source=fetch(downloads/name,entry['url'],entry['sha256'])
        subprocess.run(['dpkg-deb','-x',str(source),str(a.directory/target)],check=True)
    for entry in lock['isolated_gcc48_packages'].values():
        source=fetch(downloads/Path(entry['Filename']).name,lock['ubuntu_archive']+entry['Filename'],entry['SHA256'])
        subprocess.run(['dpkg-deb','-x',str(source),str(a.directory/'old-toolchain-root')],check=True)
    fetch(downloads/'gdev-master.tar.gz',lock['gdev']['url'],lock['gdev']['sha256'])
    print('CUDA compiler:',a.directory/'toolkit/usr/local/cuda-6.5')
    print('Isolated GCC:',a.directory/'old-toolchain-root')
    print('Gdev archive:',downloads/'gdev-master.tar.gz')

if __name__=='__main__':main()
