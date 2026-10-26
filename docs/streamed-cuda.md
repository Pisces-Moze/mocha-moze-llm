# GGUF Q4 GPU 与有界流式加载

Mocha 的 1B／1.5B GGUF 模型可把 Q4_K、Q6_K 和 Q8_0 权重矩阵乘交给 GK20A。权重以原始量化块传入 GPU，在内核中按需解码，没有整份 FP32／Q8 权重副本。RMSNorm、RoPE、注意力、KV cache 和采样仍由上游 llama.cpp CPU 路径处理。

## 与 AirLLM 的关系

参考 [AirLLM](https://github.com/lyogavin/airllm/tree/bc7ded5b7ddf798ef386735a44932c19a571f01d) 的逐层加载思想，Mocha 用原生 CUDA 桥接实现有界权重缓存和分块上传。它不是把 AirLLM Python 包直接安装到平板，也没有声称完成 PyTorch CUDA 移植。

检查的 AirLLM 提交要求 `torch>=2.4`、`transformers>=4.49,<6` 和 `accelerate>=1.0`。现有 ARMv7／sm_32 开放 Runtime 不能提供该 PyTorch 栈。AirLLM 的张量迁移和模型执行与这些框架耦合，因此本次复用已有 llama.cpp 模型解析、CPU 运算及 GGUF 文件，另外补齐原生量化 GPU 矩阵计算。未复制 AirLLM 的实现代码或安装它的训练／推理依赖。

## 使用

```sh
mocha-llm --model 1b --backend cuda --gpu-cache-mib 64 -i 'Explain Linux briefly.' -n 32
mocha-llm --model 1.5b --backend cuda --gpu-cache-mib 64 -i '请用一句话介绍Linux。' -n 32
mocha-llm --model 1.5b --backend cpu -i '你好' -n 32
```

`--gpu-cache-mib` 默认 64，可选 0／32／64／128／256 MiB。该参数限制 GPU 常驻量化权重缓存，不是全部 GPU 内存的上限；额外有 8 MiB 流式缓冲、输入输出缓冲、设备代码和驱动开销。

缓存按首次使用顺序接纳权重，满后其余矩阵按最多 8 MiB 分块传输。没有异步预取或整模型常驻的承诺。0 适合检查最小缓存流式路径；小缓存会反复上传权重，主要用于容量受限的实验。实验中的自动超大缓存和单个超大缓存池已撤回：后者运行期间管理连接失去响应，未确定根因，不能作为稳定配置。

CPU 仍可对文件映射页做缓存回收；正在执行的 GPU 缓冲不能透明换到 zram。一次只运行一个模型，更大的 GPU 缓存可能减少传输，也会减少桌面可用内存。流式加载首先解决容量，速度仍受 eMMC 读取、CPU→GPU 上传和大量小算子调度限制，不能仅凭 GPU 使用率推断加速。

GPU 错误会中止本次执行，不会把整条 CPU 推理静默标作 CUDA。成功退出时，标准错误中的 `GGUF_CUDA_STATS` 给出真实 Q4_K／Q6_K／Q8_0 调用次数、上传量、缓存命中及上传／计算时间。`--backend cpu` 不启用桥接。

诊断期间曾出现断连和用户确认的死机；重启后未保留内核故障记录，不能将其确定归因为某一算子、内存或睡眠。最后的极短功能复查在桌面服务稳定、临时阻止睡眠的条件下执行，不代表长时间稳定性。使用 GPU 实验时应保持设备唤醒；有权限的图形会话可通过 `systemd-inhibit --what=sleep --mode=block` 包住命令。SSH 会话可能需要额外授权，因此工具不会自动修改 polkit 或全局电源策略。

## 构建与检查

先按 [构建教程](build.md) 准备固定 b4514 的 llama.cpp、CUDA 6.5 编译输入和开放 Runtime。

```sh
python3 tools/patch-gguf-cuda.py build/llama.cpp
sh tools/build-gguf.sh build/llama.cpp build/gguf-armhf
sh tools/build-gguf-cuda.sh CUDA_ROOT OLD_TOOLCHAIN_ROOT RUNTIME_BUILD build/gguf-cuda
```

桥接通过 `dlopen` 加载，只作用于命令选定的进程。补丁仅接管连续、二维、固定量化权重且输入为 FP32 的矩阵乘，CPU 主线程负责 CUDA，其他 ggml 工作线程在同一屏障等待，避免跨线程使用 GPU 上下文。动态 KV 张量和不支持的算子继续由 CPU 执行。

必要的数值检查见 `tests/gguf-cuda-check.cpp`：由上游 ggml 参考函数生成并解码 Q4_K、Q6_K、Q8_0 块，再与真实 GPU 结果比较，包含批量输入、非紧密输入／输出步长及超过 8 MiB 的分块边界。该检查独立于桥接的解量化代码。CPU 正常推理会量化部分激活，因此其文本不必逐 token 与 GPU FP32 激活路径完全一致。

```sh
# 在交叉构建机编译独立检查；无需安装到系统
arm-linux-gnueabihf-g++ -O2 -std=c++11 -march=armv7-a -mfpu=neon-vfpv4 \
  -mfloat-abi=hard -mfp16-format=ieee \
  -Ibuild/llama.cpp/ggml/include -Ibuild/llama.cpp/ggml/src \
  tests/gguf-cuda-check.cpp build/gguf-armhf/build/ggml/src/libggml-base.a \
  -lm -ldl -pthread -o build/gguf-cuda/mocha-gguf-cuda-check
# 将检查程序复制到平板，指定 v0.4.0 的私有 runtime/lib 后运行
MOCHA_GGUF_GPU_CACHE_MIB=0 LD_LIBRARY_PATH=/path/to/mocha-llm-v0.4.0/runtime/lib \
  ./mocha-gguf-cuda-check
```

本桥接不等于完整 CUDA Toolkit、cuBLAS 或 PyTorch 支持，也没有修改系统 Mesa、内核或默认引导。
