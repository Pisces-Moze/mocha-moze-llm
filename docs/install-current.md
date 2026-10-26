# 安装当前版本 v0.4.0

只安装 **v0.4.0**，无需逐级升级。[版本与模型选择](versions.md) · [实测性能](performance.md)

需要 Mocha 的 ARMv7 Debian ARMhf、Python 3.11+、libstdc++6、libgomp1。CUDA 另需 Mocha 开放 Nouveau/Runtime 栈，以及当前用户的 `video`/`render` 组权限。不要把设备节点改成全局可写。

## 新安装

```sh
curl -fLo install-mocha-llm.py https://raw.githubusercontent.com/Pisces-Moze/mocha-moze-llm/main/tools/install.py
python3 install-mocha-llm.py --directory ~/Applications/mocha-llm
cd ~/Applications/mocha-llm/current
./mocha-llm --list-models
./mocha-llm --model qwen -i '用一句话介绍Linux。' -n 64
```

安装临时空间约需 1.3 GiB；最终基础权重约 502 MiB，程序另计。安装器通过固定 SHA256 下载旧基础包中的模型和当前程序，组成一个完整 v0.4.0 目录，再原子切换 `current`。不运行旧引擎，不安装 PyTorch，不修改系统驱动或图形库。

实际版本以 `current/manifest.json` 的 `version` 为准：

```sh
python3 -c 'import json; print(json.load(open("manifest.json"))["version"])'
python3 tools/verify-install.py .
```

## 已安装用户

```sh
python3 install-mocha-llm.py --directory ~/Applications/mocha-llm --reuse-from /原安装/mocha-llm-v0.3.0
```

v0.1.0 至 v0.4.0 的完整目录都可以作为权重来源，哈希必须匹配。同文件系统硬链接，跨文件系统复制。旧目录保留供回退；安装器不覆盖已有版本，不修改手工建立的全局命令链接。

离线安装：将 v0.1.0 完整归档和 v0.4.0 更新归档放在同一目录，增加 `--cache /归档目录`，仍按固定 SHA256 校验。

## 可选模型

```sh
cd ~/Applications/mocha-llm/current
python3 tools/fetch-models.py --model 1.5b --directory /有空间的数据分区/models
./mocha-llm --model 1.5b -i '用一句话介绍Linux。' -n 64
python3 tools/fetch-models.py --model 1b --directory /有空间的数据分区/models
./mocha-llm --model 1b -i 'Explain Linux briefly.' -n 64
```

下载器固定上游修订、大小和 SHA256，支持续传。可增加 `--base-url https://hf-mirror.com`，不放宽校验。1B/1.5B 权重约 770/1066 MiB，不自动下载，另需保留 256 MiB 空闲。

| 需求 | 参数 | 默认后端 |
| --- | --- | --- |
| 日常中文、响应速度 | `--model qwen` | 专用 Q8 CUDA |
| 更大中文模型 | `--model 1.5b` | NEON CPU |
| 英文指令对照 | `--model 1b` | NEON CPU |
| 最小 GPU 示例 | `--model stories` | CUDA，英文故事续写 |

Qwen 0.5B 也支持 `--backend cpu` 和 `hybrid`。大模型显式使用 `--backend cuda --gpu-cache-mib 64` 才进入实验 Q4 路径；该路径目前更慢，曾发生死机，不能视为日常加速。它与已优化的 0.5B CUDA 是不同实现。[限制与测量](streamed-cuda.md)。

## 参数与恢复

指令模型 `-n 256` 表示最多新增 256 token，提示词、模板和回答合计上限 1024。Stories 是含提示词的总步数，上限 512。模型可能提前结束。

1B/1.5B 分别需要至少 1050/1150 MiB 可用内存。交换空间不能代替 GPU/物理内存预算。GPU 缓存以外还有流式分块、激活和驱动开销。

回答写标准输出，设备初始化和统计写标准错误，可用 `2>run.log` 保留诊断。退出即释放模型；没有常驻服务。回退时运行旧目录内的命令，卸载只删除自己的安装目录、命令链接；外部 GGUF 权重自行管理。

[历史手工部署](deployment.md) 只用于重现旧版本，不是当前安装必需步骤。
