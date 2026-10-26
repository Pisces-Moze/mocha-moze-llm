// SPDX-License-Identifier: MIT
// GGUF Q4_K/Q6_K/Q8_0 decode follows ggml's MIT-licensed block definitions.
// Bounded native streaming adapter: no PyTorch, FP32 weight expansion or UVM.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cstring>
#include <ctime>
struct Entry {const void *source;unsigned bytes;unsigned char *device;};
static std::vector<Entry> entries;
static unsigned char *tile;
static float *input,*result;
static unsigned input_bytes,result_bytes,cached_bytes,budget;
static const unsigned TILE_BYTES=8U*1024*1024;
static bool initialized;
static unsigned long long calls,uploads,hits,q4_calls,q6_calls,q8_calls;
static double upload_seconds,compute_seconds;
static double now(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int ck(cudaError_t e,const char *s){if(e==cudaSuccess)return 0;std::fprintf(stderr,"GGUF_CUDA_ERROR %s: %d %s\n",s,(int)e,cudaGetErrorString(e));return 1;}
#define TRY(x) do {if(ck((x),#x))return 1;}while(0)
__device__ float half_value(unsigned short h){
    unsigned e=(h>>10)&31,m=h&1023,sign=(unsigned(h)&0x8000)<<16;
    if(!e){float f=m*5.9604644775390625e-8f;return sign?-f:f;}
    return __int_as_float((int)(sign|((e==31?255:e+112)<<23)|(m<<13)));
}
__device__ float weight(const unsigned char *row,int type,int index){
    if(type==8){const unsigned char *p=row+(index/32)*34;return half_value(*(const unsigned short *)p)*((const signed char *)(p+2))[index%32];}
    const unsigned char *p=row+(index/256)*(type==12?144:210);int i=index&255;
    if(type==12){
        int j=i/32,part=i/64;const unsigned char *s=p+4;
        unsigned scale=j<4?s[j]&63:(s[j+4]&15)|((s[j-4]>>6)<<4);
        unsigned min=j<4?s[j+4]&63:(s[j+4]>>4)|((s[j]>>6)<<4);
        unsigned q=p[16+part*32+(i&31)];q=(j&1)?q>>4:q&15;
        return half_value(*(const unsigned short *)p)*scale*q-half_value(*(const unsigned short *)(p+2))*min;
    }
    int part=i/128,g=(i&127)/32,l=i&31;
    unsigned low=p[part*64+(g&1)*32+l];low=g<2?low&15:low>>4;
    unsigned high=(p[128+part*32+l]>>(2*g))&3;
    int q=int(low|(high<<4))-32;
    int scale=((const signed char *)p)[192+part*8+g*2+l/16];
    return half_value(*(const unsigned short *)(p+208))*scale*q;
}
template<int TYPE> __device__ float quant_dot(const unsigned char *p,int cols,const float *v,int lane){
    float sum=0;
    if(TYPE==12){
        for(int base=0;base<cols;base+=256,p+=144){
            float d=half_value(*(const unsigned short *)p),dm=half_value(*(const unsigned short *)(p+2));
            const unsigned char *s=p+4;
            #pragma unroll
            for(int part=0;part<4;part++){
                int j=part*2;unsigned q=p[16+part*32+lane];
                unsigned a=j<4?s[j]&63:(s[j+4]&15)|((s[j-4]>>6)<<4);
                unsigned b=j<4?s[j+4]&63:(s[j+4]>>4)|((s[j]>>6)<<4);
                j++;unsigned c=j<4?s[j]&63:(s[j+4]&15)|((s[j-4]>>6)<<4);
                unsigned e=j<4?s[j+4]&63:(s[j+4]>>4)|((s[j]>>6)<<4);
                sum+=(d*a*(q&15)-dm*b)*v[base+part*64+lane];
                sum+=(d*c*(q>>4)-dm*e)*v[base+part*64+lane+32];
            }
        }
    }else if(TYPE==14){
        for(int base=0;base<cols;base+=256,p+=210){
            float d=half_value(*(const unsigned short *)(p+208));const signed char *sc=(const signed char *)(p+192);
            #pragma unroll
            for(int part=0;part<2;part++){
                unsigned a=p[part*64+lane],b=p[part*64+lane+32],h=p[128+part*32+lane];int s=part*8+lane/16;
                sum+=d*sc[s]*(int((a&15)|((h&3)<<4))-32)*v[base+part*128+lane];
                sum+=d*sc[s+2]*(int((b&15)|(((h>>2)&3)<<4))-32)*v[base+part*128+lane+32];
                sum+=d*sc[s+4]*(int((a>>4)|(((h>>4)&3)<<4))-32)*v[base+part*128+lane+64];
                sum+=d*sc[s+6]*(int((b>>4)|(((h>>6)&3)<<4))-32)*v[base+part*128+lane+96];
            }
        }
    }else{
        for(int base=0;base<cols;base+=32,p+=34)sum+=half_value(*(const unsigned short *)p)*((const signed char *)(p+2))[lane]*v[base+lane];
    }
    return sum;
}
template<int TYPE> __global__ void multiply(const unsigned char *w,int cols,int rows,unsigned row_bytes,const float *x,float *y){
    int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    float sum=0;const unsigned char *p=w+(unsigned)row*row_bytes;
    const float *v=x+(unsigned)blockIdx.y*cols;
    sum=quant_dot<TYPE>(p,cols,v,lane);
    for(int d=16;d;d/=2)sum+=__shfl_down(sum,d);
    if(!lane)y[(unsigned)blockIdx.y*rows+row]=sum;
}
static void finish(){
    if(!initialized)return;
    std::fprintf(stderr,"GGUF_CUDA_STATS CALLS=%llu Q4_K=%llu Q6_K=%llu Q8_0=%llu CACHE_HITS=%llu UPLOAD_BYTES=%llu CACHE_BYTES=%u TILE_BYTES=%u UPLOAD_S=%.6f COMPUTE_S=%.6f\n",calls,q4_calls,q6_calls,q8_calls,hits,uploads,cached_bytes,TILE_BYTES,upload_seconds,compute_seconds);
    for(unsigned i=0;i<entries.size();i++)cudaFree(entries[i].device);
    cudaFree(tile);cudaFree(input);cudaFree(result);
}
static int init(){
    if(initialized)return 0;
    cudaDeviceProp prop;TRY(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=3||prop.minor!=2){std::fprintf(stderr,"GGUF CUDA requires verified GK20A sm_32 runtime\n");return 1;}
    const char *value=std::getenv("MOCHA_GGUF_GPU_CACHE_MIB");int mib=value?std::atoi(value):64;
    if(mib<0||mib>256)return 1;budget=unsigned(mib)*1024*1024;
    TRY(cudaMalloc((void **)&tile,TILE_BYTES));
    initialized=true;std::atexit(finish);
    std::fprintf(stderr,"GGUF_CUDA DEVICE=%s CACHE_LIMIT_MIB=%d STREAM_TILE_MIB=8 QUANTIZED_WEIGHTS=1\n",prop.name,mib);
    return 0;
}
static int reserve(float **pointer,unsigned *capacity,unsigned bytes){
    if(bytes<=*capacity)return 0;
    if(*pointer)TRY(cudaFree(*pointer));*pointer=NULL;*capacity=0;
    TRY(cudaMalloc((void **)pointer,bytes));*capacity=bytes;return 0;
}
extern "C" int mocha_gguf_matmul(int type,const void *w,int cols,int rows,const float *x,int vectors,unsigned x_stride,float *y,unsigned y_stride){
    if((type!=8&&type!=12&&type!=14)||cols<=0||rows<=0||vectors<1||vectors>32||cols%256)return 1;
    if(init())return 1;
    unsigned row_bytes=unsigned(cols/(type==8?32:256))*(type==8?34:type==12?144:210);
    uint64_t size=uint64_t(row_bytes)*rows;if(size>0xffffffffU||row_bytes>TILE_BYTES)return 1;
    if(reserve(&input,&input_bytes,unsigned(cols)*vectors*4))return 1;
    unsigned max_rows=TILE_BYTES/row_bytes;
    const unsigned char *resident=NULL;
    for(unsigned i=0;i<entries.size();i++)if(entries[i].source==w&&entries[i].bytes==size){resident=entries[i].device;hits++;break;}
    double start=now();
    // Admission-only cache bounds memory without cycling the whole model into VRAM.
    if(!resident&&size<=budget-cached_bytes){
        unsigned char *p;TRY(cudaMalloc((void **)&p,(unsigned)size));
        cudaError_t e=cudaMemcpy(p,w,(unsigned)size,cudaMemcpyHostToDevice);
        if(e!=cudaSuccess){cudaFree(p);return ck(e,"cache upload");}
        entries.push_back(Entry{w,(unsigned)size,p});cached_bytes+=(unsigned)size;uploads+=size;resident=p;
    }
    for(int v=0;v<vectors;v++)TRY(cudaMemcpy(input+(unsigned)v*cols,(const unsigned char *)x+v*x_stride,cols*4,cudaMemcpyHostToDevice));
    upload_seconds+=now()-start;
    if(resident)max_rows=rows;
    if(reserve(&result,&result_bytes,(unsigned)(rows<max_rows?rows:max_rows)*vectors*4))return 1;
    for(unsigned offset=0;offset<(unsigned)rows;offset+=max_rows){
        unsigned n=(unsigned)rows-offset;if(n>max_rows)n=max_rows;
        const unsigned char *p=resident?resident+offset*row_bytes:tile;
        if(!resident){start=now();TRY(cudaMemcpy(tile,(const unsigned char *)w+offset*row_bytes,n*row_bytes,cudaMemcpyHostToDevice));uploads+=(uint64_t)n*row_bytes;upload_seconds+=now()-start;}
        start=now();
        if(type==12)multiply<12><<<dim3((n+3)/4,vectors),128>>>(p,cols,n,row_bytes,input,result);
        else if(type==14)multiply<14><<<dim3((n+3)/4,vectors),128>>>(p,cols,n,row_bytes,input,result);
        else multiply<8><<<dim3((n+3)/4,vectors),128>>>(p,cols,n,row_bytes,input,result);
        TRY(cudaGetLastError());TRY(cudaDeviceSynchronize());compute_seconds+=now()-start;
        for(int v=0;v<vectors;v++)TRY(cudaMemcpy((unsigned char *)y+v*y_stride+offset*4,result+(unsigned)v*n,n*4,cudaMemcpyDeviceToHost));
    }
    calls++;if(type==12)q4_calls++;else if(type==14)q6_calls++;else q8_calls++;
    return 0;
}
