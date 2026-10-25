// SPDX-License-Identifier: MIT
// Persistent Transformer CTA plus parallel vocabulary projection and selection.
// Grouped Q8 storage, warp-cooperative dot products, split-half RoPE/GQA.
#include "minimind.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <ctime>
static unsigned char *weights;
static float *cache,*output,*normalized;
static Tensor head_tensor;
struct Candidate {float value;int id;};
static Candidate *candidates;
static int *recent_tokens;
static unsigned long long calls,kernel_calls;
static double transformer_seconds,head_seconds,copy_seconds;
static double seconds(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int ck(cudaError_t e,const char *name) {
    if(e==cudaSuccess)return 0;
    std::fprintf(stderr,"CUDA %s: %d %s\n",name,(int)e,cudaGetErrorString(e));return 1;
}
#define TRY(x) do { if(ck((x),#x))return 1; } while(0)

__device__ float value(const unsigned char *b,Tensor t,int index) {
    return ((const signed char *)(b+t.data))[index]*((const float *)(b+t.scales))[index/64];
}
// Four adjacent Q8 weights per aligned load; one scale per four products.
// All supported model widths are multiples of 128.
__device__ float dot4(unsigned packed,const float *x) {
    return float(int(packed<<24)>>24)*x[0]+float(int(packed<<16)>>24)*x[1]
         +float(int(packed<<8)>>24)*x[2]+float(int(packed)>>24)*x[3];
}
__device__ float packed_dot(const unsigned char *b,Tensor t,unsigned row,const float *x) {
    unsigned lane=threadIdx.x&31;float sum=0;
    const unsigned *w=(const unsigned *)(b+t.data+row*t.cols);
    const float *sc=(const float *)(b+t.scales)+row*(t.cols/64);
    for(unsigned c=lane*4;c<t.cols;c+=128)
        sum+=dot4(w[c/4],x+c)*sc[c/64];
    for(int d=16;d;d/=2)sum+=__shfl_down(sum,d);
    return sum;
}
__device__ void matrix(const unsigned char *b,Tensor t,const float *x,float *y,bool add=false) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(unsigned row=warp;row<t.rows;row+=blockDim.x/32) {
        float sum=packed_dot(b,t,row,x);
        if(!lane) {if(add)y[row]+=sum;else y[row]=sum;}
    }
    __syncthreads();
}
__device__ void swiglu(const unsigned char *b,Tensor gate,Tensor up,const float *x,float *y) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(unsigned row=warp;row<gate.rows;row+=blockDim.x/32) {
        float a=0,z=0;
        a=packed_dot(b,gate,row,x);z=packed_dot(b,up,row,x);
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
    if(project){rms(b,h->t[TENSORS-1],x,norm,&scale);
        for(int i=tid;i<DIM;i+=blockDim.x)logits[i]=norm[i];}
}
// The large vocabulary projection has independent rows: schedule multiple CTAs.
__global__ void project_head(const unsigned char *b,Tensor t,const float *input,float *out) {
    __shared__ float x[DIM];
    for(int i=threadIdx.x;i<DIM;i+=blockDim.x)x[i]=input[i];
    __syncthreads();
    unsigned row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    for(;row<t.rows;row+=gridDim.x*(blockDim.x/32)) {
        float sum=packed_dot(b,t,row,x);if(!(threadIdx.x&31))out[row]=sum;
    }
}
__global__ void latency_probe(float *out){if(!threadIdx.x)out[0]=1;}
static int profile_latency(){
    if(!std::getenv("MOCHA_LLM_PROFILE"))return 0;
    for(int i=0;i<3;i++){double a=seconds();latency_probe<<<1,32>>>(output);kernel_calls++;TRY(cudaDeviceSynchronize());
        double b=seconds();float v;TRY(cudaMemcpy(&v,output,4,cudaMemcpyDeviceToHost));
        std::fprintf(stderr,"CUDA_LATENCY LAUNCH_S=%.6f READ4_S=%.6f VALUE=%g\n",b-a,seconds()-b,v);}
    return 0;
}
// Partition the vocabulary into 64 disjoint ranges. Keep k candidates from
// each range; their union necessarily contains the global top k. Repetition
// penalties and control-token masking must precede selection.
__global__ void select_candidates(const float *logits,Candidate *out,const int *recent,int count,float penalty,int k,int rows) {
    __shared__ float best[128];__shared__ int ids[128],history[64];
    int tid=threadIdx.x;if(tid<count)history[tid]=recent[tid];__syncthreads();
    float values[19];int indexes[19];
    #pragma unroll
    for(int j=0;j<19;j++) {
        int id=blockIdx.x*128+tid+j*64*128;indexes[j]=id;float v=id<rows?logits[id]:-FLT_MAX;
        if(id<rows){for(int h=0;h<count;h++)if(history[h]==id){v=v<0?v*penalty:v/penalty;break;}
#ifdef MOCHA_QWEN
            if(id>=151643&&id!=EOS)v=-1e30f;
#else
            if(id<2)v=-1e30f;
#endif
        }
        values[j]=v;
    }
    for(int rank=0;rank<k;rank++) {
        float v=-FLT_MAX;int id=0x7fffffff;
        #pragma unroll
        for(int j=0;j<19;j++)if(values[j]>v||(values[j]==v&&indexes[j]<id)){v=values[j];id=indexes[j];}
        best[tid]=v;ids[tid]=id;__syncthreads();
        for(int d=64;d;d/=2){if(tid<d&&(best[tid+d]>best[tid]||(best[tid+d]==best[tid]&&ids[tid+d]<ids[tid]))){best[tid]=best[tid+d];ids[tid]=ids[tid+d];}__syncthreads();}
        if(!tid){out[blockIdx.x*k+rank].value=best[0];out[blockIdx.x*k+rank].id=ids[0];}
        int winner=ids[0];__syncthreads();
        #pragma unroll
        for(int j=0;j<19;j++)if(indexes[j]==winner)values[j]=-FLT_MAX;
    }
}
extern "C" int gpu_init(const void *b,unsigned bytes) {
    cudaDeviceProp p;TRY(cudaGetDeviceProperties(&p,0));if(p.major!=3||p.minor!=2)return 1;
    TRY(cudaMalloc((void **)&weights,bytes));TRY(cudaMalloc((void **)&cache,2*LAYERS*CONTEXT*KV*4));
    TRY(cudaMalloc((void **)&output,VOCAB*4));TRY(cudaMalloc((void **)&normalized,DIM*4));
    TRY(cudaMalloc((void **)&candidates,64*64*sizeof(Candidate)));TRY(cudaMalloc((void **)&recent_tokens,64*sizeof(int)));
    TRY(cudaMemcpy(weights,b,bytes,cudaMemcpyHostToDevice));head_tensor=((const Header *)b)->t[0];
    std::fprintf(stderr,"BACKEND=cuda DEVICE=%s Q8_GROUP=64 GPU_WEIGHTS=%u GPU_KV=%u\n",p.name,bytes,2*LAYERS*CONTEXT*KV*4);
    return profile_latency();
}
extern "C" int gpu_forward(float *logits,int token,int pos,int project,int readback) {
    const char *option=std::getenv("MOCHA_LLM_CUDA_THREADS");int threads=option?std::atoi(option):512;
    if(threads!=256&&threads!=512&&threads!=1024)return 1;
    if(threads<DIM/2)return 1;
    bool profile=std::getenv("MOCHA_LLM_PROFILE")!=NULL;double start=profile?seconds():0,mid=start,end=start;
    forward<<<1,threads>>>(weights,cache,normalized,token,pos,project);TRY(cudaGetLastError());
    if(profile){TRY(cudaDeviceSynchronize());mid=seconds();transformer_seconds+=mid-start;}
    if(project){project_head<<<64,128>>>(weights,head_tensor,normalized,output);TRY(cudaGetLastError());}
    TRY(cudaDeviceSynchronize());
    if(profile){end=seconds();head_seconds+=end-mid;}
    if(project&&readback)TRY(cudaMemcpy(logits,output,VOCAB*4,cudaMemcpyDeviceToHost));
    if(profile)copy_seconds+=seconds()-end;
    calls++;kernel_calls+=1+(project?1:0);return 0;
}
extern "C" int gpu_collect(float *logits,const int *history,int count,float penalty,int k,int rows) {
    if(k<1||k>64||count<0||count>64)return 1;
    TRY(cudaMemcpy(recent_tokens,history,count*sizeof(int),cudaMemcpyHostToDevice));
    select_candidates<<<64,128>>>(output,candidates,recent_tokens,count,penalty,k,rows);TRY(cudaGetLastError());
    kernel_calls++;
    TRY(cudaDeviceSynchronize());Candidate result[4096];
    TRY(cudaMemcpy(result,candidates,64*k*sizeof(Candidate),cudaMemcpyDeviceToHost));
    for(int i=0;i<rows;i++)logits[i]=-FLT_MAX;
    for(int i=0;i<64*k;i++)if(result[i].id>=0&&result[i].id<rows)logits[result[i].id]=result[i].value;
    return 0;
}
extern "C" int gpu_head_init(const void *b) {
    cudaDeviceProp p;TRY(cudaGetDeviceProperties(&p,0));if(p.major!=3||p.minor!=2)return 1;
    Tensor original=((const Header *)b)->t[0];head_tensor=original;
    unsigned size=original.rows*original.cols;head_tensor.data=0;head_tensor.scales=size;
    TRY(cudaMalloc((void **)&weights,size+size/16));
    TRY(cudaMalloc((void **)&output,VOCAB*4));TRY(cudaMalloc((void **)&normalized,DIM*4));
    TRY(cudaMalloc((void **)&candidates,64*64*sizeof(Candidate)));TRY(cudaMalloc((void **)&recent_tokens,64*sizeof(int)));
    TRY(cudaMemcpy(weights,(const unsigned char *)b+original.data,size,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(weights+size,(const unsigned char *)b+original.scales,size/16,cudaMemcpyHostToDevice));
    std::fprintf(stderr,"BACKEND=hybrid DEVICE=%s GPU_WEIGHTS=%u\n",p.name,size+size/16);
    return 0;
}
extern "C" int gpu_head_start(const float *input,int rows) {
    TRY(cudaMemcpy(normalized,input,DIM*4,cudaMemcpyHostToDevice));
    Tensor t=head_tensor;t.rows=rows;
    project_head<<<64,128>>>(weights,t,normalized,output);TRY(cudaGetLastError());calls++;kernel_calls++;
    return 0;
}
extern "C" int gpu_head_finish(float *logits,int rows,int readback) {
    TRY(cudaDeviceSynchronize());if(readback)TRY(cudaMemcpy(logits,output,rows*4,cudaMemcpyDeviceToHost));return 0;
}
extern "C" int gpu_finish() {
    if(std::getenv("MOCHA_LLM_PROFILE"))std::fprintf(stderr,"CUDA_PROFILE TRANSFORMER_S=%.6f HEAD_S=%.6f COPY_S=%.6f\n",transformer_seconds,head_seconds,copy_seconds);
    if(candidates)TRY(cudaFree(candidates));if(recent_tokens)TRY(cudaFree(recent_tokens));
    TRY(cudaFree(normalized));TRY(cudaFree(output));if(cache)TRY(cudaFree(cache));TRY(cudaFree(weights));
    std::fprintf(stderr,"CUDA_FORWARD_STEPS=%llu CUDA_KERNEL_CALLS=%llu\n",calls,kernel_calls);return calls?0:1;
}
