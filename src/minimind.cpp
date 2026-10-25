// SPDX-License-Identifier: MIT
// Host driver and independent CPU reference for the grouped-Q8 CUDA model.
#include "minimind.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
static const unsigned char *base;
static const Header *header;
static std::vector<float> keys(LAYERS*CONTEXT*KV),values(keys.size());
static float val(Tensor t,int i) {return ((const signed char *)(base+t.data))[i]*((const float *)(base+t.scales))[i/64];}
static void mat(Tensor t,const float *x,float *y,bool add=false,unsigned first=0) {
    const signed char *w=(const signed char *)(base+t.data);const float *sc=(const float *)(base+t.scales);
    #pragma omp parallel for schedule(static)
    for(unsigned r=first;r<t.rows;r++) {
        float sum=0;
        for(unsigned g=0;g<t.cols/64;g++) {
            int off=r*t.cols+g*64;float dot=0;
#ifdef __ARM_NEON
            float32x4_t accum=vdupq_n_f32(0);
            for(int j=0;j<64;j+=8) {
                int16x8_t q=vmovl_s8(vld1_s8(w+off+j));
                accum=vmlaq_f32(accum,vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))),vld1q_f32(x+g*64+j));
                accum=vmlaq_f32(accum,vcvtq_f32_s32(vmovl_s16(vget_high_s16(q))),vld1q_f32(x+g*64+j+4));
            }
            float tmp[4];vst1q_f32(tmp,accum);dot=tmp[0]+tmp[1]+tmp[2]+tmp[3];
#else
            for(int j=0;j<64;j++)dot+=w[off+j]*x[g*64+j];
#endif
            sum+=dot*sc[off/64];
        }
        if(add)y[r]+=sum;else y[r]=sum;
    }
}
static void rms(Tensor t,const float *x,float *y) {
    float sum=0;for(int i=0;i<DIM;i++)sum+=x[i]*x[i];float s=1/std::sqrt(sum/DIM+RMS_EPS);
    for(int i=0;i<DIM;i++)y[i]=x[i]*s*((const float *)(base+t.data))[i];
}
static void cpu(float *out,int token,int pos,bool project=true,bool hidden_only=false) {
    float x[DIM],n[DIM],q[DIM],k[KV],v[KV],a[CONTEXT],gate[HID],up[HID];
    for(int i=0;i<DIM;i++)x[i]=val(header->t[0],token*DIM+i);
    for(int l=0;l<LAYERS;l++) {
        int t=1+l*STRIDE;rms(header->t[t],x,n);mat(header->t[t+1],n,q);mat(header->t[t+2],n,k);mat(header->t[t+3],n,v);
#ifdef MOCHA_QWEN
        for(int i=0;i<DIM;i++)q[i]+=((const float *)(base+header->t[t+9].data))[i];
        for(int i=0;i<KV;i++){k[i]+=((const float *)(base+header->t[t+10].data))[i];v[i]+=((const float *)(base+header->t[t+11].data))[i];}
#endif
        for(int h=0;h<DIM/HEAD;h++)for(int i=0;i<32;i++) {
            float angle=pos*std::pow(1000000.f,-2.f*i/64),cs=std::cos(angle),sn=std::sin(angle);
            int j=h*64+i;float z=q[j],b=q[j+32];q[j]=z*cs-b*sn;q[j+32]=b*cs+z*sn;
            if(h<2) {z=k[j];b=k[j+32];int o=l*CONTEXT*KV+pos*KV+j;
                keys[o]=z*cs-b*sn;keys[o+32]=b*cs+z*sn;}
        }
        for(int i=0;i<KV;i++)values[l*CONTEXT*KV+pos*KV+i]=v[i];
        for(int h=0;h<DIM/HEAD;h++) {
            float largest=-1e30f,total=0;
            for(int s=0;s<=pos;s++) {float z=0;for(int i=0;i<64;i++)z+=q[h*64+i]*keys[l*CONTEXT*KV+s*KV+(h/(DIM/KV))*64+i];
                a[s]=z*.125f;largest=std::max(largest,a[s]);}
            for(int s=0;s<=pos;s++){a[s]=std::exp(a[s]-largest);total+=a[s];}
            for(int i=0;i<64;i++) {float z=0;for(int s=0;s<=pos;s++)z+=(a[s]/total)*values[l*CONTEXT*KV+s*KV+(h/(DIM/KV))*64+i];n[h*64+i]=z;}
        }
        mat(header->t[t+4],n,x,true);rms(header->t[t+8],x,n);mat(header->t[t+5],n,gate);mat(header->t[t+6],n,up);
        for(int i=0;i<HID;i++)gate[i]=(gate[i]/(1+std::exp(-gate[i])))*up[i];
        mat(header->t[t+7],gate,x,true);
    }
    if(project){rms(header->t[TENSORS-1],x,n);
        if(hidden_only)std::memcpy(out,n,sizeof(n));else mat(header->t[0],n,out);}
}
static double now() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char **argv) {
    if(argc!=11) {std::fprintf(stderr,"usage: engine MODEL TOKEN_IDS NEW_TOKENS cpu|cuda|hybrid TEMPERATURE SEED CHECK REPEAT_PENALTY TOP_K TOP_P\n");return 2;}
    int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))return 2;
    base=(const unsigned char *)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(base==MAP_FAILED)return 2;
    header=(const Header *)base;
    if(st.st_size<HEADER_BYTES||header->magic!=0x384D4D4D||header->version!=1||header->bytes!=(unsigned)st.st_size||
       header->dim!=DIM||header->hidden!=HID||header->layers!=LAYERS||header->vocab!=VOCAB||header->group!=64)return 2;
    for(int i=0;i<TENSORS;i++) {Tensor t=header->t[i];uint64_t size=(uint64_t)t.rows*(t.scales?t.cols:4);
        if(t.data<HEADER_BYTES||t.data+size>(uint64_t)st.st_size||
           (t.scales&&(t.cols%64||t.scales+size/64*4>(uint64_t)st.st_size)))return 2;}
    std::vector<int> prompt;FILE *f=std::fopen(argv[2],"r");if(!f)return 2;int token;
    while(std::fscanf(f,"%d",&token)==1) {if(token<0||token>=VOCAB)return 2;prompt.push_back(token);}std::fclose(f);
    int count=std::atoi(argv[3]);float temp=std::atof(argv[5]);unsigned seed=std::strtoul(argv[6],NULL,10);
    float penalty=std::atof(argv[8]),top_p=std::atof(argv[10]);int top_k=std::atoi(argv[9]);
    if(prompt.empty()||count<1||prompt.size()+count>CONTEXT||temp<0||!std::isfinite(temp)||
       !std::isfinite(penalty)||penalty<1||top_k<1||top_k>VOCAB||!(top_p>0&&top_p<=1))return 2;
    bool use_gpu=!std::strcmp(argv[4],"cuda"),hybrid=!std::strcmp(argv[4],"hybrid"),check=std::atoi(argv[7]);
    if(!use_gpu&&!hybrid&&std::strcmp(argv[4],"cpu"))return 2;
    const char *percent_env=std::getenv("MOCHA_LLM_GPU_HEAD_PERCENT");int percent=percent_env?std::atoi(percent_env):100;
    if(percent<1||percent>100)return 2;int gpu_rows=VOCAB*percent/100;
    bool sparse=(use_gpu||hybrid)&&!check&&top_k<=64&&!std::getenv("MOCHA_LLM_FULL_LOGITS");
    bool verify_candidates=use_gpu&&sparse&&std::getenv("MOCHA_LLM_VERIFY_CANDIDATES");
    double start=now();if(use_gpu&&gpu_init(base,st.st_size))return 3;
    if(hybrid&&gpu_head_init(base))return 3;
    if(use_gpu&&!check)madvise((void *)base,st.st_size,MADV_DONTNEED);
    float logits[VOCAB],reference[VOCAB],max_error=0;unsigned long compared=0;
    double ready=now(),decode_start=0;int generated=0;token=prompt[0];std::vector<int> history=prompt;
    for(int pos=0;pos<(int)prompt.size()+count-1;pos++) {
        // Prompt-only steps update KV; their vocabulary projection is unused.
        bool project=check||pos+1>=(int)prompt.size();
        if(hybrid) {
            float hidden[DIM];cpu(hidden,token,pos,project,true);
            if(project){int error=0;
                // Keep CUDA on its owning host thread. Another OpenMP worker
                // computes disjoint vocabulary rows while the GPU is active.
                if(gpu_rows<VOCAB){int workers=1;
                    #pragma omp parallel num_threads(2) shared(error,workers)
                    {
                        if(omp_get_thread_num()==0){workers=omp_get_num_threads();error=gpu_head_start(hidden,gpu_rows);
                            if(!error)error=gpu_head_finish(logits,gpu_rows,!sparse);}
                        else mat(header->t[0],hidden,logits,false,gpu_rows);
                    }
                    if(workers==1)mat(header->t[0],hidden,logits,false,gpu_rows);
                }else{error=gpu_head_start(hidden,gpu_rows);if(!error)error=gpu_head_finish(logits,gpu_rows,!sparse);}
                if(error)return 3;
                if(sparse){int kept=std::min(64,(int)history.size());
                    if(gpu_collect(logits,history.data()+history.size()-kept,kept,penalty,temp==0?1:top_k,gpu_rows))return 3;}}
        }
        if(use_gpu) {
            if(gpu_forward(logits,token,pos,project,!sparse||verify_candidates))return 3;
            if(project&&sparse){int kept=std::min(64,(int)history.size());
                int k=temp==0?1:top_k;
                if(verify_candidates)std::memcpy(reference,logits,sizeof(logits));
                if(gpu_collect(logits,history.data()+history.size()-kept,kept,penalty,k,VOCAB))return 3;
                if(verify_candidates){bool visited[VOCAB]={false};
                    for(int j=history.size()-kept;j<(int)history.size();j++){int id=history[j];
                        if(!visited[id]){reference[id]=reference[id]<0?reference[id]*penalty:reference[id]/penalty;visited[id]=true;}}
#ifdef MOCHA_QWEN
                    for(int id=151643;id<VOCAB;id++)if(id!=EOS)reference[id]=-1e30f;
#else
                    reference[0]=reference[1]=-1e30f;
#endif
                    std::vector<int> order(VOCAB);for(int i=0;i<VOCAB;i++)order[i]=i;
                    std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](int a,int b){return reference[a]>reference[b]||(reference[a]==reference[b]&&a<b);});
                    for(int j=0;j<k;j++){int id=order[j];if(!std::isfinite(logits[id])||std::fabs(logits[id]-reference[id])>1e-5f*(1+std::fabs(reference[id]))){
                        std::fprintf(stderr,"CANDIDATE CHECK FAILED pos=%d id=%d value=%g reference=%g\n",pos,id,logits[id],reference[id]);return 4;}}
                }}
        }
        if((use_gpu||hybrid)&&check) {cpu(reference,token,pos);for(int i=0;i<VOCAB;i++) {
                float d=std::fabs(logits[i]-reference[i]);max_error=std::max(d,max_error);compared++;
                if(!std::isfinite(logits[i])||d>0.003f*(1+std::fabs(reference[i]))) {
                    std::fprintf(stderr,"CHECK FAILED pos=%d logit=%d actual=%g reference=%g\n",pos,i,logits[i],reference[i]);return 4;}
            }}
        if(!use_gpu&&!hybrid)cpu(logits,token,pos,project);
        if(pos+1<(int)prompt.size()) {token=prompt[pos+1];continue;}
        if(!decode_start)decode_start=now();
        bool seen[VOCAB]={false};
        for(int j=std::max(0,(int)history.size()-64);j<(int)history.size();j++) {
            int id=history[j];if(sparse&&(!hybrid||id<gpu_rows))continue;
            if(!seen[id]){logits[id]=logits[id]<0?logits[id]*penalty:logits[id]/penalty;seen[id]=true;}
        }
        // Model-specific control tokens; EOS is left available for sampling.
#ifdef MOCHA_QWEN
        for(int id=151643;id<VOCAB;id++)if(id!=EOS)logits[id]=-1e30f;
#else
        logits[0]=logits[1]=-1e30f;
#endif
        if(temp==0)token=std::max_element(logits,logits+VOCAB)-logits;
        else {std::vector<int> order(VOCAB);for(int i=0;i<VOCAB;i++)order[i]=i;
            std::partial_sort(order.begin(),order.begin()+top_k,order.end(),[&](int a,int b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);});
            float largest=logits[order[0]],sum=0;
            for(int j=0;j<top_k;j++){int i=order[j];logits[i]=std::exp((logits[i]-largest)/temp);sum+=logits[i];}
            float cutoff=0;int kept=0;while(kept<top_k&&cutoff<sum*top_p)cutoff+=logits[order[kept++]];
            seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;float r=(seed/4294967296.0)*cutoff;token=order[kept-1];
            for(int j=0;j<kept;j++){int i=order[j];r-=logits[i];if(r<=0){token=i;break;}}
        }
        if(token==EOS)break;
        std::printf("%d\n",token);std::fflush(stdout);generated++;history.push_back(token);
    }
    double done=now();
    std::fprintf(stderr,"BACKEND=%s PROMPT_TOKENS=%zu GENERATED=%d STARTUP_S=%.3f PREFILL_S=%.3f DECODE_TPS=%.3f WALL_S=%.3f CHECKED=%lu MAX_ABS_ERROR=%.8g\n",
        hybrid?"hybrid":use_gpu?"cuda":"cpu",prompt.size(),generated,ready-start,(decode_start?decode_start:done)-ready,
        generated>1?(generated-1)/(done-decode_start):0,done-start,compared,max_error);
    unsigned long rss=0,peak=0;char line[256];FILE *status=std::fopen("/proc/self/status","r");
    if(status){while(std::fgets(line,sizeof(line),status)) {std::sscanf(line,"VmRSS: %lu kB",&rss);std::sscanf(line,"VmHWM: %lu kB",&peak);}std::fclose(status);}
    std::fprintf(stderr,"RSS_KIB=%lu PEAK_RSS_KIB=%lu\n",rss,peak);
    int result=(use_gpu||hybrid)?gpu_finish():0;munmap((void *)base,st.st_size);return result;
}
