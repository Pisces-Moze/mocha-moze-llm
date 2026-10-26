// SPDX-License-Identifier: MIT
// Independent CPU dequantization comes from pinned upstream ggml, not the bridge.
#include "ggml.h"
#include "ggml-quants.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
typedef int (*Mat)(int,const void *,int,int,const float *,int,unsigned,float *,unsigned);
static int check(Mat mul,int type,int cols,int rows,int vectors){
    unsigned row=unsigned(ggml_row_size((ggml_type)type,cols));
    std::vector<unsigned char> packed(row*rows);
    std::vector<float> source(cols),decoded(cols*3),x((cols+8)*vectors),y((rows+7)*vectors,321.f);
    for(int p=0;p<3;p++){
        for(int i=0;i<cols;i++)source[i]=std::sin(i*.37+p)*.17+std::cos(i*.11)*.09;
        void *q=&packed[p*row];
        if(type==12){quantize_row_q4_K_ref(source.data(),(block_q4_K *)q,cols);dequantize_row_q4_K((const block_q4_K *)q,&decoded[p*cols],cols);}
        else if(type==14){quantize_row_q6_K_ref(source.data(),(block_q6_K *)q,cols);dequantize_row_q6_K((const block_q6_K *)q,&decoded[p*cols],cols);}
        else{quantize_row_q8_0_ref(source.data(),(block_q8_0 *)q,cols);dequantize_row_q8_0((const block_q8_0 *)q,&decoded[p*cols],cols);}
    }
    for(int r=3;r<rows;r++)std::memcpy(&packed[r*row],&packed[(r%3)*row],row);
    for(int v=0;v<vectors;v++)for(int c=0;c<cols;c++)x[v*(cols+8)+c]=std::cos(c*.23+v)*.21;
    if(mul(type,packed.data(),cols,rows,x.data(),vectors,(cols+8)*4,y.data(),(rows+7)*4))return 1;
    float max_error=0;
    for(int v=0;v<vectors;v++)for(int r=0;r<rows;r++){
        float expected=0;for(int c=0;c<cols;c++)expected+=decoded[(r%3)*cols+c]*x[v*(cols+8)+c];
        float actual=y[v*(rows+7)+r],error=std::fabs(actual-expected);max_error=std::fmax(error,max_error);
        if(!std::isfinite(actual)||error>.002f*(1+std::fabs(expected))){std::fprintf(stderr,"CHECK_FAIL type=%d r=%d v=%d actual=%g expected=%g\n",type,r,v,actual,expected);return 1;}
    }
    for(int v=0;v<vectors;v++)for(int i=rows;i<rows+7;i++)if(y[v*(rows+7)+i]!=321.f)return 1;
    std::printf("Q4_GPU_CHECK type=%d cols=%d rows=%d vectors=%d max_error=%g\n",type,cols,rows,vectors,max_error);
    return 0;
}
int main(){
    void *lib=dlopen("libmocha-gguf-cuda.so",RTLD_NOW|RTLD_LOCAL);if(!lib){std::fprintf(stderr,"%s\n",dlerror());return 1;}
    Mat mul=(Mat)dlsym(lib,"mocha_gguf_matmul");if(!mul)return 1;
    for(int type:{12,14,8})if(check(mul,type,512,17,3))return 1;
    if(check(mul,12,4096,4097,2))return 1;
    return 0;
}
