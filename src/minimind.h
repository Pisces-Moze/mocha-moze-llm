// SPDX-License-Identifier: MIT
#ifndef MOCHA_MINIMIND_H
#define MOCHA_MINIMIND_H
#include <stdint.h>
struct Tensor { uint32_t data, scales, rows, cols; };
#ifdef MOCHA_QWEN
enum { DIM=896, HID=4864, KV=128, HEAD=64, VOCAB=151936, CONTEXT=256, LAYERS=24,
       STRIDE=12, TENSORS=290, BOS=151644, EOS=151645, HEADER_BYTES=8192 };
#define RMS_EPS 1e-6f
#else
enum { DIM=512, HID=1408, KV=128, HEAD=64, VOCAB=6400, CONTEXT=512, LAYERS=8,
       STRIDE=9, TENSORS=74, BOS=1, EOS=2, HEADER_BYTES=4096 };
#define RMS_EPS 1e-5f
#endif
struct Header { uint32_t magic,version,bytes,dim,hidden,layers,vocab,group; Tensor t[TENSORS]; };
extern "C" int gpu_init(const void *,unsigned);
extern "C" int gpu_forward(float *,int,int,int,int);
extern "C" int gpu_finish();
extern "C" int gpu_head_init(const void *);
extern "C" int gpu_head_start(const float *,int);
extern "C" int gpu_head_finish(float *,int,int);
extern "C" int gpu_collect(float *,const int *,int,float,int,int);
#endif
