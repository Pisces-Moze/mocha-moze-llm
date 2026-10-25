// SPDX-License-Identifier: MIT
// New MiniMind2-Small inference implementation: one persistent CTA per token.
// Grouped Q8 storage, warp-cooperative dot products, split-half RoPE/GQA.
#include "minimind.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>
static unsigned char *weights;
static float *cache,*output;
static unsigned long long calls;
static int ck(cudaError_t e,const char *name) {
    if(e==cudaSuccess)return 0;
    std::fprintf(stderr,"CUDA %s: %d %s\n",name,(int)e,cudaGetErrorString(e));return 1;
}
#define TRY(x) do { if(ck((x),#x))return 1; } while(0)

__device__ float value(const unsigned char *b,Tensor t,int index) {
    return ((const signed char *)(b+t.data))[index]*((const float *)(b+t.scales))[index/64];
}
__device__ void matrix(const unsigned char *b,Tensor t,const float *x,float *y,bool add=false) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(unsigned row=warp;row<t.rows;row+=blockDim.x/32) {
        float sum=0;
        for(unsigned col=lane;col<t.cols;col+=32)sum+=value(b,t,row*t.cols+col)*x[col];
        for(int d=16;d;d/=2)sum+=__shfl_down(sum,d);
        if(!lane) {if(add)y[row]+=sum;else y[row]=sum;}
    }
    __syncthreads();
}
__device__ void swiglu(const unsigned char *b,Tensor gate,Tensor up,const float *x,float *y) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(unsigned row=warp;row<gate.rows;row+=blockDim.x/32) {
        float a=0,z=0;
        for(unsigned c=lane;c<gate.cols;c+=32) {
            float v=x[c];a+=value(b,gate,row*gate.cols+c)*v;z+=value(b,up,row*up.cols+c)*v;
        }
        for(int d=16;d;d/=2){a+=__shfl_down(a,d);z+=__shfl_down(z,d);}
        if(!lane)y[row]=__fdividef(a,1+__expf(-a))*z;
    }
    __syncthreads();
}
__device__ void rms(const unsigned char *b,Tensor t,const float *x,float *y,float *scale) {
    if(!threadIdx.x) {float sum=0;for(int i=0;i<DIM;i++)sum+=x[i]*x[i];*scale=rsqrtf(sum/DIM+RMS_EPS);}
    __syncthreads();
    for(int i=threadIdx.x;i<DIM;i+=blockDim.x)y[i]=x[i]*(*scale)*((const float *)(b+t.data))[i];
    __syncthreads();
}
__global__ void forward(const unsigned char *b,float *kv,float *logits,int token,int pos,int project) {
    const Header *h=(const Header *)b;int tid=threadIdx.x;
    __shared__ float x[DIM],norm[DIM],q[DIM],k[KV],v[KV],gate[HID],scale;
    for(int i=tid;i<DIM;i+=blockDim.x)x[i]=value(b,h->t[0],token*DIM+i);__syncthreads();
    float *keys=kv,*values=kv+LAYERS*CONTEXT*KV;
    for(int l=0;l<LAYERS;l++) {
        int t=1+l*STRIDE;
        rms(b,h->t[t],x,norm,&scale);
        matrix(b,h->t[t+1],norm,q);matrix(b,h->t[t+2],norm,k);matrix(b,h->t[t+3],norm,v);
#ifdef MOCHA_QWEN
        for(int i=tid;i<DIM;i+=blockDim.x)q[i]+=((const float *)(b+h->t[t+9].data))[i];
        if(tid<KV){k[tid]+=((const float *)(b+h->t[t+10].data))[tid];v[tid]+=((const float *)(b+h->t[t+11].data))[tid];}
        __syncthreads();
#endif
        if(tid<DIM/2) {
            int head=tid/32,i=tid%32,idx=head*64+i;
            float sn,cs;__sincosf(pos*__powf(1000000.f,-2.f*i/64),&sn,&cs);
            float a=q[idx],z=q[idx+32];q[idx]=a*cs-z*sn;q[idx+32]=z*cs+a*sn;
            if(head<2) {a=k[idx];z=k[idx+32];int off=l*CONTEXT*KV+pos*KV+idx;
                keys[off]=a*cs-z*sn;keys[off+32]=z*cs+a*sn;}
        }
        if(tid<KV)values[l*CONTEXT*KV+pos*KV+tid]=v[tid];
        __syncthreads();
        // One warp handles each attention head, including coalesced KV reads.
        int head=tid/32,lane=tid&31;
        if(head<DIM/HEAD) {
            // Online softmax avoids an entire heads*context shared buffer.
            float largest=-FLT_MAX,total=0,z0=0,z1=0;
            for(int s=0;s<=pos;s++) {
                int off=l*CONTEXT*KV+s*KV+(head/(DIM/KV))*64;
                float dot=q[head*64+lane]*keys[off+lane]+q[head*64+lane+32]*keys[off+lane+32];
                for(int d=16;d;d/=2)dot+=__shfl_down(dot,d);
                dot=__shfl(dot,0)*.125f;
                float next=fmaxf(largest,dot),old=__expf(largest-next),weight=__expf(dot-next);
                z0=z0*old+weight*values[off+lane];z1=z1*old+weight*values[off+lane+32];
                total=total*old+weight;largest=next;
            }
            norm[head*64+lane]=__fdividef(z0,total);norm[head*64+lane+32]=__fdividef(z1,total);
        }
        __syncthreads();
        matrix(b,h->t[t+4],norm,x,true);
        rms(b,h->t[t+8],x,norm,&scale);
        swiglu(b,h->t[t+5],h->t[t+6],norm,gate);matrix(b,h->t[t+7],gate,x,true);
    }
    if(project){rms(b,h->t[TENSORS-1],x,norm,&scale);matrix(b,h->t[0],norm,logits);}
}
extern "C" int gpu_init(const void *b,unsigned bytes) {
    cudaDeviceProp p;TRY(cudaGetDeviceProperties(&p,0));if(p.major!=3||p.minor!=2)return 1;
    TRY(cudaMalloc((void **)&weights,bytes));TRY(cudaMalloc((void **)&cache,2*LAYERS*CONTEXT*KV*4));
    TRY(cudaMalloc((void **)&output,VOCAB*4));TRY(cudaMemcpy(weights,b,bytes,cudaMemcpyHostToDevice));
    std::fprintf(stderr,"BACKEND=cuda DEVICE=%s Q8_GROUP=64 GPU_WEIGHTS=%u GPU_KV=%u\n",p.name,bytes,2*LAYERS*CONTEXT*KV*4);
    return 0;
}
extern "C" int gpu_forward(float *logits,int token,int pos,int project) {
    const char *option=std::getenv("MOCHA_LLM_CUDA_THREADS");int threads=option?std::atoi(option):512;
    if(threads!=256&&threads!=512&&threads!=1024)return 1;
    if(threads<DIM/2)return 1;
    forward<<<1,threads>>>(weights,cache,output,token,pos,project);TRY(cudaGetLastError());
    TRY(cudaDeviceSynchronize());
    if(project)TRY(cudaMemcpy(logits,output,VOCAB*4,cudaMemcpyDeviceToHost));
    calls++;return 0;
}
extern "C" int gpu_finish() {
    TRY(cudaFree(output));TRY(cudaFree(cache));TRY(cudaFree(weights));
    std::fprintf(stderr,"CUDA_KERNEL_CALLS=%llu\n",calls);return calls?0:1;
}
