#!/usr/bin/env python3
"""Convert the pinned MiniMind2-Small F16 checkpoint to grouped symmetric Q8.

Requires NumPy on the build computer only. This is a new storage format;
it is neither GGUF nor a format accepted by NVIDIA's official runtime.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

SOURCE_SHA = '83bfe6f127c98120a3410aab65eee3b66b8301ac352c20b7efd5e2eb688f6d85'
MATRICES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def convert(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Checkpoint hash does not match pinned MiniMind2-Small')
    length = struct.unpack_from('<Q', raw)[0]
    header = json.loads(raw[8:8 + length])
    body = memoryview(raw)[8 + length:]
    data = bytearray(4096)
    descriptors = []

    def tensor(name, quantized=True):
        info = header[name + '.weight']
        assert info['dtype'] == 'F16'
        start, end = info['data_offsets']
        array = np.frombuffer(body[start:end], dtype='<f2').astype('<f4').reshape(info['shape'])
        offset = len(data)
        if quantized:
            groups = array.reshape(-1, 64)
            scale = np.maximum(np.max(np.abs(groups), axis=1) / 127, 1e-12).astype('<f4')
            quant = np.rint(groups / scale[:, None]).clip(-127, 127).astype('i1')
            data.extend(quant.tobytes())
            scales = len(data)
            data.extend(scale.tobytes())
        else:
            data.extend(array.tobytes())
            scales = 0
        descriptors.append((offset, scales, array.shape[0], array.shape[-1]))

    tensor('model.embed_tokens')
    for layer in range(8):
        base = f'model.layers.{layer}.'
        tensor(base + 'input_layernorm', False)
        for name in MATRICES:
            tensor(base + name)
        tensor(base + 'post_attention_layernorm', False)
    tensor('model.norm', False)
    assert len(descriptors) == 74
    struct.pack_into('<8I', data, 0, 0x384D4D4D, 1, len(data), 512, 1408, 8, 6400, 64)
    for i, descriptor in enumerate(descriptors):
        struct.pack_into('<4I', data, 32 + i * 16, *descriptor)
    output.write_bytes(data)
    print(json.dumps({'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                      'parameters': 25829888, 'format': 'group64-q8-f32-scales'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
