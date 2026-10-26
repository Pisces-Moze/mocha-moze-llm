#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attach an opt-in quantized matmul bridge to pinned llama.cpp b4514."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);a=p.parse_args()
target=a.source/'ggml/src/ggml-cpu/ggml-cpu.c';text=target.read_text()
if 'MOCHA_GGUF_CUDA_BRIDGE' in text:
    start=text.index('    // MOCHA_GGUF_CUDA_BRIDGE')
    end=text.index('    enum ggml_type           const vec_dot_type',start)
    text=text[:start]+text[end:]
if 'MOCHA_GGUF_CUDA_BRIDGE' not in text:
    if '#include <dlfcn.h>' not in text:text='#include <dlfcn.h>\n'+text
    start=text.index('static void ggml_compute_forward_mul_mat(')
    at=text.index('    enum ggml_type           const vec_dot_type',start)
    insert='''    // MOCHA_GGUF_CUDA_BRIDGE: attention/norm/sampling remain upstream CPU.
    const char * mocha_cuda = getenv("MOCHA_GGUF_CUDA");
    if (mocha_cuda && strcmp(mocha_cuda, "1") == 0 &&
        (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q8_0) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ne00 % 256 == 0 && ne11 <= 32 && ggml_is_contiguous(src0) && strstr(src0->name, ".weight") &&
        nb10 == sizeof(float) && nb0 == sizeof(float)) {
        if (ith == 0) {
            typedef int (*mocha_mul_t)(int,const void *,int,int,const float *,int,unsigned,float *,unsigned);
            static mocha_mul_t mocha_mul;
            if (!mocha_mul) {
                void * lib = dlopen("libmocha-gguf-cuda.so", RTLD_NOW | RTLD_LOCAL);
                if (!lib) GGML_ABORT("Mocha GGUF CUDA bridge load failed: %s", dlerror());
                mocha_mul = (mocha_mul_t)dlsym(lib, "mocha_gguf_matmul");
                if (!mocha_mul) GGML_ABORT("Mocha GGUF CUDA bridge symbol missing");
            }
            if (mocha_mul(src0->type,src0->data,ne00,ne01,src1->data,ne11,nb11,dst->data,nb1))
                GGML_ABORT("Mocha GGUF CUDA computation failed; no silent CPU fallback");
        }
        ggml_barrier(params->threadpool);
        return;
    }

'''
    text=text[:at]+insert+text[at:];target.write_text(text)
print('Patched optional GGUF quantized CUDA matmul')
