#!/usr/bin/env python3
"""Download the pinned original Qwen weights for local conversion."""
import argparse
import json
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    spec=spec_from_file_location('prepare_toolchain',repo/'tools/prepare-toolchain.py')
    tools=module_from_spec(spec);spec.loader.exec_module(tools)
    lock=json.loads((repo/'models/qwen-inputs.json').read_text())
    url=lock['repository']+'/resolve/'+lock['revision']+'/model.safetensors'
    tools.fetch(a.directory/'model.safetensors',url,lock['source_sha256'])
    print(a.directory/'model.safetensors')

if __name__=='__main__':main()
