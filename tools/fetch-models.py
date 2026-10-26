#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download optional GGUF weights pinned by revision, size and SHA256."""
import argparse,hashlib,json,os,shutil,time,urllib.request
from pathlib import Path

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    models=json.loads((root/'models/large-models.json').read_text(encoding='utf-8'))['models']
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=list(models)+['1b','1.5b','all'])
    p.add_argument('--directory',type=Path,help='external model directory; create a symlink under this release')
    p.add_argument('--base-url',help='optional HTTPS mirror, e.g. https://hf-mirror.com; SHA256 remains mandatory')
    a=p.parse_args();key={'1b':'llama1b','1.5b':'qwen1.5b'}.get(a.model,a.model)
    folder=(a.directory or root/'models').resolve();folder.mkdir(parents=True,exist_ok=True)
    for key,m in models.items() if key=='all' else [(key,models[key])]:
        dest=folder/m['filename'];partial=dest.with_suffix(dest.suffix+'.partial')
        if dest.exists():
            if dest.stat().st_size!=m['bytes'] or digest(dest)!=m['sha256']:p.error('Existing model is corrupt: '+str(dest))
        else:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>m['bytes']:p.error('Partial download is larger than expected: '+str(partial))
            if shutil.disk_usage(folder).free < m['bytes']-offset+256*1024*1024:p.error('Insufficient disk space; choose --directory on another filesystem (256 MiB reserve required)')
            url=m['url']
            if a.base_url:
                if not a.base_url.startswith('https://'):p.error('Mirror must use HTTPS')
                url=a.base_url.rstrip('/')+'/'+m['repository']+'/resolve/'+m['revision']+'/'+m['filename']
            for attempt in range(4):
                try:
                    offset=partial.stat().st_size if partial.exists() else 0
                    headers={'User-Agent':'Mocha-LLM/0.3.0'}
                    if offset:headers['Range']=f'bytes={offset}-'
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as src:
                        append=offset and src.status==206
                        if append and not src.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Unexpected download range')
                        with partial.open('ab' if append else 'wb') as out:
                            while block:=src.read(1024*1024):out.write(block)
                    break
                except (OSError,ValueError):
                    if attempt==3:raise
                    time.sleep(2)
            if partial.stat().st_size!=m['bytes'] or digest(partial)!=m['sha256']:p.error('Downloaded model failed SHA256 verification: '+str(partial))
            partial.replace(dest)
        link=root/'models'/m['filename']
        if link.absolute()!=dest:
            if link.is_symlink():
                if link.resolve()!=dest:p.error('Model link points elsewhere: '+str(link))
            elif link.exists():p.error('Release already has a separate model file: '+str(link))
            else:link.symlink_to(dest)
        print('Verified and installed:',key,dest)
if __name__=='__main__':main()
