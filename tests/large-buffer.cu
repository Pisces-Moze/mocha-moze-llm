// SPDX-License-Identifier: MIT
// GPU generic loads must not alias the legacy 224/240 MiB apertures.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
static void ck(cudaError_t result) {
    if(result!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(result));std::exit(1);}
}
__global__ void sample(const unsigned *base,unsigned *out,unsigned pages) {
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<pages;i+=blockDim.x*gridDim.x) {
        unsigned address=(unsigned)(base+i*1024),value;
        // Force generic addressing: global-only loads can hide aperture collisions.
        asm volatile("ld.u32 %0, [%1];" : "=r"(value) : "r"(address));
        out[i]=value;
    }
}
int main() {
    const unsigned bytes=512U*1024*1024,chunk=1024*1024,pages=bytes/4096;
    unsigned *buffer,*output;
    std::vector<unsigned> pattern(chunk/4),readback(pages);
    for(unsigned i=0;i<pattern.size();i++)pattern[i]=0x4d4f4348U^i;
    ck(cudaMalloc((void **)&buffer,bytes));ck(cudaMalloc((void **)&output,pages*4));
    for(unsigned offset=0;offset<bytes;offset+=chunk)
        ck(cudaMemcpy((unsigned char *)buffer+offset,&pattern[0],chunk,cudaMemcpyHostToDevice));
    sample<<<32,256>>>(buffer,output,pages);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(&readback[0],output,pages*4,cudaMemcpyDeviceToHost));
    unsigned errors=0;
    for(unsigned i=0;i<pages;i++)if(readback[i]!=pattern[(i*1024)%(chunk/4)])errors++;
    ck(cudaFree(output));ck(cudaFree(buffer));
    std::printf("GPU_GENERIC_BUFFER_BYTES=%u PAGE_SAMPLES=%u ERRORS=%u\n",bytes,pages,errors);
    return errors?1:0;
}
