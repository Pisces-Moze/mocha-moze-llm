#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert pinned Qwen2.5-0.5B-Instruct BF16 to the Mocha grouped-Q8 format."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

SOURCE_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
MATRICES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

def convert(source, output):
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SOURCE_SHA:
            raise ValueError('Source checkpoint SHA256 mismatch')
    with source.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    tensors = []
    with source.open('rb') as source_stream, output.open('wb') as dest:
        dest.write(bytes(8192))
        def tensor(name, quantized=True, bias=False):
            info = header[name + ('.bias' if bias else '.weight')]
            assert info['dtype'] == 'BF16'
            start, end = info['data_offsets']
            source_stream.seek(8 + length + start)
            raw = source_stream.read(end - start)
            data = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4')
            offset = dest.tell()
            if quantized:
                groups = data.reshape(-1, 64)
                scales = np.maximum(np.abs(groups).max(axis=1) / 127, 1e-12).astype('<f4')
                # Bound temporary quantization arrays for the large embedding.
                for first in range(0, len(groups), 16384):
                    last = min(first + 16384, len(groups))
                    dest.write(np.rint(groups[first:last] / scales[first:last, None]).clip(-127, 127).astype('i1').tobytes())
                scale_offset = dest.tell()
                dest.write(scales.tobytes())
            else:
                dest.write(data.tobytes())
                scale_offset = 0
            tensors.append((offset, scale_offset, info['shape'][0], info['shape'][-1]))
        tensor('model.embed_tokens')
        for layer in range(24):
            base = f'model.layers.{layer}.'
            tensor(base + 'input_layernorm', False)
            for name in MATRICES:
                tensor(base + name)
            tensor(base + 'post_attention_layernorm', False)
            for name in ['q_proj', 'k_proj', 'v_proj']:
                tensor(base + 'self_attn.' + name, False, True)
        tensor('model.norm', False)
        assert len(tensors) == 290
        size = dest.tell()
        dest.seek(0)
        dest.write(struct.pack('<8I', 0x384D4D4D, 1, size, 896, 4864, 24, 151936, 64))
        for t in tensors:
            dest.write(struct.pack('<4I', *t))
    with output.open('rb') as stream:
        print(json.dumps({'bytes': size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    convert(a.checkpoint, a.output)
