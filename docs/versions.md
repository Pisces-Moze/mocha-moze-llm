# 安装哪个版本

新用户只安装 **v0.4.0**。运行 `tools/install.py` 会得到一个完整目录，不需要逐个升级，也不需要保留 v0.1.0 程序。内部下载旧基础包只为取得经过 SHA256 校验的两个小模型权重。

| 版本 | 增量功能 | 新用户是否需要单独安装 |
| --- | --- | --- |
| v0.1.0 | 两个小模型与初始开放 CUDA Runtime | 否，安装器只提取其权重 |
| v0.2.0 / v0.2.1 | Qwen 0.5B 的 CUDA 优化、1024 上下文、CPU/协同路径 | 否，已包含在 v0.4.0 |
| v0.3.0 | 可选 Llama 1B、Qwen 1.5B，NEON CPU GGUF 引擎 | 否，已包含在 v0.4.0 |
| **v0.4.0** | 保留以上功能，增加实验 Q4_K/Q6_K/Q8_0 GPU 桥接 | **当前完整功能版本** |

## 模型与后端怎么选

| 需求 | 参数 | 默认执行方式 |
| --- | --- | --- |
| 日常中文、优先响应速度 | `--model qwen` | Qwen 0.5B，专用 Q8 CUDA |
| 更大的中文模型 | `--model 1.5b` | Qwen 1.5B，NEON CPU |
| 英文 1B 对照 | `--model 1b` | Llama 1B，NEON CPU |
| 最小 GPU 连通性示例 | `--model stories` | TinyStories CUDA，英文故事续写 |
| 研究量化 GPU 计算 | `--model 1.5b --backend cuda --gpu-cache-mib 64` | 实验 Q4 GPU + CPU 注意力，当前更慢 |

1B/1.5B 权重不随基础安装下载，使用 `tools/fetch-models.py` 按需安装。Q4 CUDA 曾出现需要重启的死机，尚不适合日常使用；这与已优化的 0.5B CUDA 是不同路径。AirLLM 的流式思想已用于原生桥接，**没有移植完整 PyTorch/AirLLM**。

`current` 链接只指向安装器最终校验成功的 v0.4.0 目录。查看 `current/manifest.json` 的 `version` 可确认实际版本；`--list-models` 查看权重和后端。旧包保留在 Releases 供重现历史结果，不是需要依次安装的依赖链。

已有用户：`python3 tools/install.py --reuse-from /现有/mocha-llm-v0.3.0`，复用三个已校验的小模型文件。不会自动切换系统全局命令；若以前手工设置了 `/usr/local/bin/mocha-llm`，请将它指向新目录。
