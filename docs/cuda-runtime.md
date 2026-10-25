# 大模型需要的 Gdev Runtime 修复

Gdev Nouveau 路径历史上将 generic shared/local 窗口设为 `0x0e000000` 和 `0x0f000000`。小型 CUDA 程序通常没有分配到这些地址；Qwen 的约 501 MiB 连续权重缓冲会覆盖该区间，generic 读指令因此可能读到 shared/local 空间。仅从 CPU 映射读回权重不能证明 GPU 读到相同内容。

实机诊断中，前几层结果接近 CPU，之后部分权重读为零，第一步 logit 对照失败。独立 GPU 页抽样也能检测到错误。这不是 Qwen 量化误差，也不应通过增大对照容差绕过。

`runtime-source/apply-runtime-port.py` 在 GK20A 上将窗口改为 `0xe0000000` / `0xf0000000`，远离本机实际分配的大缓冲。其他芯片保留原配置。这是针对 Mocha 2 GiB / ARMv7 的后端修正，不是通用的 64 位虚拟地址分配器；更大地址空间或其他 GPU 需要重新设计和验证。

修复后的诊断对 525,128,192 字节权重每页抽取一项，共 128,205 个样本，没有不一致；真实 Qwen 进一步通过 6,229,376 个 CPU／GPU logits 对照。独立 512 MiB 回归覆盖 131,072 个页样本：旧 Runtime 有 4096 个错误，新 Runtime 为零。源码位于 `tests/large-buffer.cu`，构建和运行见 [构建教程](build.md)。

发行包把开放 Runtime 放在自己的 `runtime/lib` 中，通过进程环境加载。它保留既有 ARMhf Driver ABI、ELF32 cubin 解析、NVCC 参数布局、真实内存填充、事件与函数属性修复。未实现的功能应返回错误；本工具不提供 cuBLAS、PyTorch CUDA 或完整 NVIDIA CUDA 兼容承诺。

## v0.2.0 提交与映射开销

仅 GK20A 的 Gdev 计算缓冲增加 `NOUVEAU_BO_COHERENT`。ARM CPU 缓存与 GPU 不一致时，普通永久映射的缓存 BO 在每次提交、读回时付出整块缓存维护成本；本次大权重下空内核即超过一秒。coherent 映射把这项开销移出热路径，同时保留所有 fence 等待、BO 驻留验证和 CPU/GPU 所有权检查。CPU 侧读取 coherent 映射仍较慢，因此词表候选在 GPU 选择后小批量回传。

静态属性查询复用线程本地已验证的 GK20A 芯片身份；实际 SM 数仍走硬件查询，其他芯片仍使用原流程。不缓存频率，不伪造异步能力。Ocelot 的原生 kernel launch 仍同步；hybrid 使用主线程调用 CUDA，另一 OpenMP 工作者计算互不重叠的 CPU 词表行，实现实际并行。

背景资料：[Nouveau 的 ARM coherent/WC 映射修正](https://kernel.googlesource.com/pub/scm/linux/kernel/git/westeri/thunderbolt/+/774b73428e6eabb4f0382aeeb76e569c7b106a29%5E%21/)。本工具没有用关闭同步或缓存一致性保护换取性能。
