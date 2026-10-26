# Mocha LLM

[代码许可：MIT](LICENSE) · [部署教程](docs/deployment.md) · [实机性能](docs/performance.md) · [源码构建](docs/build.md)

在小米平板 Mocha（Tegra K1、ARMv7）上运行本地语言模型，使用参数切换微型故事、Qwen 0.5B、Llama 1B 和 Qwen 1.5B。0.5B 提供 CUDA／CPU／协同路径，新增大模型使用 llama.cpp 的 ARMv7 NEON CPU 量化推理。

这是针对 Mocha 开放驱动栈的实验工具。CUDA 路径执行真实 GPU 前向计算；速度取决于模型和开放 Runtime，不能据此认为 GPU 一定快于 CPU。

## 安装与运行

预编译包见 [Releases](https://github.com/Pisces-Moze/mocha-moze-llm/releases)。基础包包含两个小模型，更新包增加 ARMhf 程序、私有开放 Runtime 和许可证；1B／1.5B 权重按需从固定上游版本下载。不需要在平板上安装编译器或 Python 推理框架。

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
cd ..
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.3.0/mocha-llm-v0.3.0-armhf-update.tar.gz
curl -fLo SHA256SUMS-v0.3.0 https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.3.0/SHA256SUMS
sha256sum -c SHA256SUMS-v0.3.0
tar -xzf mocha-llm-v0.3.0-armhf-update.tar.gz
cd mocha-llm-v0.3.0
python3 tools/reuse-models.py --from ../mocha-llm-v0.1.0
./mocha-llm --model qwen -i '请用一句话介绍Linux。' -n 64
```

回答逐步输出到终端，运行统计写入标准错误。v0.2.1 默认使用 CUDA。优化后的中文 CUDA 实测约 7.75 token/s，初始化约 3.2 秒、34-token 提示词预填充约 2.6 秒；v0.1.0 的 CUDA 约 0.67 token/s。CPU 与 GPU 使用相同 Q8 权重。CPU 四线程此前约 3.09 token/s，本轮连续测试末尾约 2.40 token/s，结果受温度、调频和桌面负载影响。

```sh
# 中文对话：有完整指令模板，默认使用 Qwen
./mocha-llm --backend cpu -i '请写一句简短的中文问候语。' -n 64

# 更高 CPU 吞吐，运行时可能影响桌面响应
./mocha-llm --backend cpu --threads 4 -i '用一句话介绍Linux。' -n 48 -t 0

# 实际 CUDA 前向计算；v0.2.1 默认后端
./mocha-llm --backend cuda -i '用一句话介绍Linux。' -n 16 -t 0

# 英文微型模型：用于 CUDA 连通性和数值测试
./mocha-llm --model stories --backend cuda -i 'Once upon a time' -n 96 -t 0
```

## 升级到 v0.2.1

历史版本 v0.2.1 也可从 v0.1.0 模型包升级：[小型更新包](https://github.com/Pisces-Moze/mocha-moze-llm/releases/tag/v0.2.1)。模型权重不变，更新无需重复下载约 501 MiB 的 Qwen 文件。旧版升级命令见 [部署教程](docs/deployment.md#升级到-v021)；当前安装入口使用上面的 v0.3.0。

CPU＋GPU 协同模式可运行 `./mocha-llm --backend hybrid --threads 4 --gpu-head-percent 90 -i "你好" -n 32`。实测约 3.61 token/s，当前比纯 CUDA 慢，保留用于调整分工；不作为默认。

v0.2.1 将 Qwen 总上下文从 256 扩到 1024 token，修复 `-n 256` 被提示词和模板挤占预算的问题。超限时会明确显示当前提示词下允许的最大 `-n`。`-n` 是新生成 token 的上限，遇到结束标记仍会提前停止。

## v0.3.0：1B／1.5B 模型切换

在现有安装旁边解压 [v0.3.0 更新包](https://github.com/Pisces-Moze/mocha-moze-llm/releases/tag/v0.3.0)，运行 `python3 tools/reuse-models.py --from ../mocha-llm-v0.2.1` 复用原权重；新安装也可从 v0.1.0 基础包复用。然后按需安装大模型：

```sh
python3 tools/fetch-models.py --model 1b
python3 tools/fetch-models.py --model 1.5b
./mocha-llm --list-models
./mocha-llm --model 1b -i 'Explain Linux in one sentence.' -n 64
./mocha-llm --model 1.5b -i '请用一句话介绍Linux。' -n 64
./mocha-llm --model qwen --backend cuda -i '你好' -n 64
```

两份新增权重合计约 1.79 GiB。每份下载后都验证 SHA256，并要求目标分区保留至少 256 MiB。空间不足时用 `--directory /path/on/another/filesystem` 分开放置，工具自动建立模型链接；中国大陆可加 `--base-url https://hf-mirror.com`，哈希校验仍强制执行。

大模型默认选择 CPU，显式指定 CUDA 会报出支持范围。当前开放 CUDA 引擎读取专用 Q8 格式，不能直接读取 GGUF。1B 更适合英文，中文优先使用 Qwen 1.5B。一次只加载被选中的模型；程序退出后释放其缓存。完整升级、空间规划与模型卸载见 [部署教程](docs/deployment.md#v030大模型)。

2026-10-27 实机短回答验证：1B 约 1.18 token/s、峰值 RSS 842 MiB；1.5B 约 0.95 token/s、峰值 RSS 1127 MiB。两者均保持桌面运行、使用默认三线程；加载和首轮回答分别合计约 49／53 秒。结果是短时功能检查，详见 [记录与测量口径](docs/performance.md#v030大模型)。

## 模型

| 模型 | 用途 | 权重 | 本工具上下文 |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 中文问答、简单指令 | 分组 Q8，约 501 MiB | 1024 token |
| TinyStories 260K | 英文故事续写、CUDA 最小示例 | FP32，约 1 MiB | 512 token |
| Llama 3.2 1B Instruct | 英文指令、较大模型对照 | GGUF Q4_K_M，约 770 MiB | 1024 token；NEON CPU |
| Qwen2.5 1.5B Instruct | 中文问答、较大模型 | GGUF Q4_K_M，约 1066 MiB | 1024 token；NEON CPU |

TinyStories 没有指令训练，把问题接成故事属于模型自身限制。Qwen 能跟随中文指令，但仍可能产生错误，不具备联网检索能力。

Qwen 的 1024-token 上限是这个移植版本的内存和延迟取舍，不是原模型的上限。当前只提供单次问答；每条命令重新建立模型和 KV cache，不自动保留历史。

v0.2.1 的 CUDA 初始化与上述提示词预填充合计约 5.8 秒；等待期间没有文本输出属于预期。具体耗时随提示词长度和系统负载变化。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | `qwen` | `qwen`、`stories`、`1b`（`llama1b`）、`1.5b`（`qwen1.5b`）；`qwen0.5b` 是旧 Qwen 的别名 |
| `--list-models` | 关闭 | 列出模型、支持后端及安装状态 |
| `--backend` | 按模型选择 | 小模型默认 `cuda`，1B／1.5B 默认 `cpu`；`hybrid` 仅用于 Qwen 0.5B，后端不匹配时明确报错 |
| `--gpu-head-percent` | `90` | hybrid 的 GPU 词表行比例，可选 50／75／90／100；CPU 并行处理其余词表行 |
| `-i` | 简短示例 | 提示词，中文模型自动套用对应 ChatML 模板 |
| `-n` | `96` | 指令模型：最多新生成 token；Stories：包含提示词的总推理步数 |
| `-t` | `0.7` | 温度；`0` 为确定性贪心选择 |
| `--repeat-penalty` | `1.12` | 中文模型最近 64 token 的重复惩罚 |
| `--top-k` / `--top-p` | `40` / `0.9` | 中文模型采样范围；温度为 0 时不使用 |
| `--seed` | `42` | 采样种子 |
| `--threads` | `3` | CPU／CPU 对照的线程数，范围 1–4 |
| `--check` | 关闭 | 每一步 CUDA logits 与独立 CPU 路径比较，额外耗时和内存 |

`token` 不等于汉字个数。Qwen 的提示词、模板和输出合计必须不超过 1024 token。Stories 保留原引擎的 `-n` 含义，其重复惩罚和 Top-k／Top-p 参数不生效。

## 兼容性与性能

验证平台是 Mocha、GK20A（`sm_32`）、ARMhf Debian 和原生 Nouveau。需要 Python 3.11+、系统 glibc/libstdc++/libgomp，以及当前用户能访问相应 `/dev/dri` 节点。它不是 x86 或 ARM64 的通用安装包，也不能直接用于原厂 Android。

运行库只在应用进程的 `LD_LIBRARY_PATH` 中加载，不替换系统 Mesa、Nouveau 或 NVIDIA 库。无需刷写引导器、重启、锁定频率或提高电压。

实测数据、测量口径和 GPU 限制见 [性能记录](docs/performance.md)。部署、故障处理和卸载见 [部署教程](docs/deployment.md)。重建方法见 [构建说明](docs/build.md)。

0.5B 的专用引擎使用 24 层、分组 Q8 权重、GQA、RoPE、QKV bias 和 ChatML；CPU 矩阵乘使用 NEON，CUDA 在 GK20A 上执行完整 Transformer 前向。预填充跳过不用于采样的词表投影。发行版开放 Runtime 修正了旧 Gdev 的 generic shared/local 地址窗口与大权重冲突，真实 Qwen CPU／CUDA 对照已覆盖 6,229,376 个 logits。

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
- [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/ec7f3ac9ab33e46b136eb5ab6a76c4d81f57c7f1)：MIT，固定 b4514，用于 ARMv7 CPU GGUF 引擎。
- [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)：Apache-2.0。
- [Llama 3.2 1B Instruct GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)：Meta Llama 3.2 Community License 与 Acceptable Use Policy，**不是 MIT**。Built with Llama。来源、固定版本、量化和哈希见 [大模型清单](models/large-models.json)。

不分发 NVIDIA CUDA Toolkit、NVCC、专有 libcudart 或供应商头文件。此工具不能代表完整 NVIDIA CUDA 运行时、PyTorch CUDA 或 cuBLAS 已经可用。
