# Mocha LLM

[代码许可：MIT](LICENSE) · [部署教程](docs/deployment.md) · [实机性能](docs/performance.md) · [源码构建](docs/build.md)

在小米平板 Mocha（Tegra K1、ARMv7）上运行本地语言模型，提供中文指令模型、英文微型故事模型，以及可检查的 CUDA／NEON CPU 推理路径。

这是针对 Mocha 开放驱动栈的实验工具。CUDA 路径执行真实 GPU 前向计算；速度取决于模型和开放 Runtime，不能据此认为 GPU 一定快于 CPU。

## 安装与运行

预编译包见 [Releases](https://github.com/Pisces-Moze/mocha-moze-llm/releases)。包内包含两个模型、ARMhf 程序、私有开放 Runtime 和许可证，不需要在平板上安装编译器或 Python 推理框架。

```sh
# 在平板终端执行
mkdir -p ~/Applications/mocha-llm
cd ~/Applications/mocha-llm
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.1.0/mocha-llm-v0.1.0-armhf.tar.gz
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.1.0/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf mocha-llm-v0.1.0-armhf.tar.gz
cd mocha-llm-v0.1.0
python3 tools/verify-install.py .
./mocha-llm --model qwen --backend cpu -i '请用一句话介绍Linux。' -n 64
```

回答逐步输出到终端，运行统计写入标准错误。第一次调用会初始化 GPU，首字等待包含初始化和提示词处理时间。中文模型实测 CPU 三线程约 2.52 token/s、四线程约 3.09 token/s，CUDA 约 0.67 token/s；日常默认三线程 CPU，为桌面留出余量。需要最高实测 CPU 吞吐时使用 `--backend cpu --threads 4`。两个后端使用相同 Q8 权重。

```sh
# 中文对话：有完整指令模板，默认使用 Qwen
./mocha-llm --backend cpu -i '请写一句简短的中文问候语。' -n 64

# 更高 CPU 吞吐，运行时可能影响桌面响应
./mocha-llm --backend cpu --threads 4 -i '用一句话介绍Linux。' -n 48 -t 0

# 实际 CUDA 前向计算；此平台 GPU 路径目前更慢
./mocha-llm --backend cuda -i '用一句话介绍Linux。' -n 16 -t 0

# 英文微型模型：用于 CUDA 连通性和数值测试
./mocha-llm --model stories --backend cuda -i 'Once upon a time' -n 96 -t 0
```

## 两个模型

| 模型 | 用途 | 权重 | 本工具上下文 |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 中文问答、简单指令 | 分组 Q8，约 501 MiB | 256 token |
| TinyStories 260K | 英文故事续写、CUDA 最小示例 | FP32，约 1 MiB | 512 token |

TinyStories 没有指令训练，把问题接成故事属于模型自身限制。Qwen 能跟随中文指令，但仍可能产生错误，不具备联网检索能力。

Qwen 的 256-token 上限是这个移植版本的内存和延迟取舍，不是原模型的上限。当前只提供单次问答；每条命令重新建立模型和 KV cache，不自动保留历史。

本次 34-token 提示词的 CPU 首字等待约 10 秒，CUDA 初始化与预填充合计约 52 秒；等待期间没有文本输出属于预期。具体耗时随提示词长度和系统负载变化。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | `qwen` | `qwen` 或 `stories` |
| `--backend` | `cpu` | GPU 或 NEON CPU；默认选择实测更快的 CPU，CUDA 失败会报错，不静默换后端 |
| `-i` | 简短示例 | 提示词，中文模型自动套用对应 ChatML 模板 |
| `-n` | `96` | Qwen：最多新生成 token；Stories：包含提示词的总推理步数 |
| `-t` | `0.7` | 温度；`0` 为确定性贪心选择 |
| `--repeat-penalty` | `1.12` | 中文模型最近 64 token 的重复惩罚 |
| `--top-k` / `--top-p` | `40` / `0.9` | 中文模型采样范围；温度为 0 时不使用 |
| `--seed` | `42` | 采样种子 |
| `--threads` | `3` | CPU／CPU 对照的线程数，范围 1–4 |
| `--check` | 关闭 | 每一步 CUDA logits 与独立 CPU 路径比较，额外耗时和内存 |

`token` 不等于汉字个数。Qwen 的提示词、模板和输出合计必须不超过 256 token。Stories 保留原引擎的 `-n` 含义，其重复惩罚和 Top-k／Top-p 参数不生效。

## 兼容性与性能

验证平台是 Mocha、GK20A（`sm_32`）、ARMhf Debian 和原生 Nouveau。需要 Python 3.11+、系统 glibc/libstdc++/libgomp，以及当前用户能访问相应 `/dev/dri` 节点。它不是 x86 或 ARM64 的通用安装包，也不能直接用于原厂 Android。

运行库只在应用进程的 `LD_LIBRARY_PATH` 中加载，不替换系统 Mesa、Nouveau 或 NVIDIA 库。无需刷写引导器、重启、锁定频率或提高电压。

实测数据、测量口径和 GPU 限制见 [性能记录](docs/performance.md)。部署、故障处理和卸载见 [部署教程](docs/deployment.md)。重建方法见 [构建说明](docs/build.md)。

Qwen 使用 24 层、分组 Q8 权重、GQA、RoPE、QKV bias 和 ChatML；CPU 矩阵乘使用 NEON，CUDA 在 GK20A 上执行完整 Transformer 前向。预填充跳过不用于采样的词表投影。发行版开放 Runtime 修正了旧 Gdev 的 generic shared/local 地址窗口与大权重冲突，真实 Qwen CPU／CUDA 对照已覆盖 6,229,376 个 logits。

## 源码与验证

```sh
python3 -m unittest discover -s tests -v
```

分词器测试采用 Hugging Face tokenizers 生成的固定参考值，覆盖中文、空白、组合字符和对话边界。CUDA 数值检查需要真实 Mocha，普通 CI 只运行不依赖 GPU 的测试。

仓库提供 [可选 GitHub Actions 模板](ci/source-check.yml.example)，目前未启用。发布所用 GitHub 凭据缺少 `workflow` 权限；有相应权限的维护者可将模板放入 `.github/workflows/check.yml`。上面的本地检查不依赖 Actions。

`src/minimind.cpp`／`.cu` 是早期 MiniMind 适配扩展得到的通用两种固定形状引擎；正式中文构建使用 `MOCHA_QWEN`。MiniMind 开发参考保留在源码中，发行包不安装其权重。

## 来源与许可

集成代码采用 MIT；各上游材料保留各自许可证，见 [NOTICE](NOTICE) 和 [licenses](licenses)。

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：Apache-2.0，固定版本与哈希见 [模型清单](models/qwen-inputs.json)。
- [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) 和 [llama2.c](https://github.com/karpathy/llama2.c)：MIT，固定输入见 [故事模型清单](models/stories-inputs.json)。
- [Gdev](https://github.com/shinpei0208/gdev)：MIT，开放 Runtime 的 Mocha 修改和上游固定输入位于 [runtime-source](runtime-source)。

不分发 NVIDIA CUDA Toolkit、NVCC、专有 libcudart 或供应商头文件。此工具不能代表完整 NVIDIA CUDA 运行时、PyTorch CUDA 或 cuBLAS 已经可用。
