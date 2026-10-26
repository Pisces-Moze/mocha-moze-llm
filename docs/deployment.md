# 部署教程

> 历史手工部署记录。新用户请使用 [当前 v0.4.0 统一安装教程](install-current.md)，无需依次安装下面的各个版本。版本差异见 [选择表](versions.md)。

## 1. 确认系统

在平板执行：

```sh
uname -m
python3 --version
ls -l /dev/dri
grep MemAvailable /proc/meminfo
```

发行包面向 `armv7l` 的 Debian ARMhf。Python 需为 3.11 或更新版本；Qwen 运行时建议至少有 700 MiB 可用内存，CPU／CUDA 同时数值对照建议至少 1.1 GiB。关闭不用的浏览器标签页和重型程序，再运行模型。无需停止桌面。

若程序报告 `libgomp.so.1` 缺失，安装 Debian 的 `libgomp1`。若缺少 C++ 运行库，安装 `libstdc++6`。通过系统既有软件源安装；中国大陆构建环境可使用阿里云 Debian 镜像，勿混用不同 Debian 发行版的软件源。

CUDA 后端需要已启用的 GK20A Nouveau 和用户访问权限：

```sh
id
```

通常用户应在 `video`／`render` 组中。若管理员新增了组，需要重新登录。不要为了运行这个程序把设备节点改成全局可写。

## 2. 下载并校验

按根 README 的完整下载命令解压到数据分区或用户目录。解压后运行：

```sh
python3 tools/verify-install.py .
```

它逐项检查大小和 SHA256。不要把模型文件提交到自己的 Git 历史；版本权重已随 Release 发布。权重不需要再次从 Hugging Face 下载。

## 3. 中文问答

```sh
./mocha-llm --model qwen --backend cuda -i '用两句话解释什么是操作系统。' -n 64
./mocha-llm --model qwen --backend cpu --threads 3 -i '请写一句欢迎语。' -n 48
```

不需要手写 `<|im_start|>`。工具负责模板和分词；用户输入中的特殊标记按普通文本处理。`-n` 是上限，模型发出结束标记会提前退出。

需要可复现输出时使用 `-t 0`。若要比较完全未经重复惩罚的贪心路径，再加 `--repeat-penalty 1`。较小模型也可能重复或答错，增加温度不等于提升知识准确率。

## 4. GPU 数值检查

```sh
./mocha-llm --model qwen --backend cuda --check -i '你好' -n 8 -t 0 2>cuda-check.log
```

检查发生在采样前，覆盖全部词表 logits。日志中 `CHECKED` 应大于 0；若出现 `CHECK FAILED` 或 CUDA 错误，命令非零退出，不会伪装成 GPU 成功。

`[gdev] Opened gdev0`／`Closed gdev0` 是底层设备初始化输出；它们不是模型回答。完整诊断保留在标准错误，需要只显示回答时可像上例保存到日志。不要直接丢弃错误日志，以免看不到故障。

## 5. 建议参数

v0.2.1 日常默认 CUDA，当前实测快于 CPU 和混合后端。需要释放 GPU 给桌面时可以选择三线程 CPU。`hybrid` 由 CPU 处理 Transformer、GPU 处理大部分词表，CPU 另一个工作线程同时处理剩余词表；当前约 3.61 token/s，低于纯 CUDA。混合模式建议至少有 850 MiB 可用内存。

如需比较 CUDA 的线程块，可在单次命令前设置：

```sh
MOCHA_LLM_CUDA_THREADS=1024 ./mocha-llm --backend cuda -i '你好' -n 32
```

Qwen 默认 1024，也支持 512；实测 1024 稍快。此参数不会修改 GPU 频率。不要把不同温度、提示词长度或 `--check` 状态的速度直接比较。

## 6. 卸载和恢复

退出命令会释放 GPU 权重和 KV cache。没有常驻服务、系统启动项或全局图形库替换。

如果按 README 解压到 `~/Applications/mocha-llm`，删除自己解压出的 `mocha-llm-v0.1.0` 文件夹即可卸载该版本。若手动建立过命令软链接，再删除对应软链接。已有系统 CUDA 实验目录和旧模型不受影响。

## 升级到 v0.2.1

先保留 v0.1.0 完整目录。在它的父目录执行：

```sh
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.2.1/mocha-llm-v0.2.1-armhf-update.tar.gz
curl -fLo SHA256SUMS-v0.2.1 https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.2.1/SHA256SUMS
sha256sum -c SHA256SUMS-v0.2.1
tar -xzf mocha-llm-v0.2.1-armhf-update.tar.gz
cd mocha-llm-v0.2.1
python3 tools/reuse-models.py --from ../mocha-llm-v0.1.0
./mocha-llm -i '请用一句话介绍Linux。' -n 64
```

复用工具先校验旧模型哈希，再在同一文件系统内硬链接；跨文件系统则复制。旧目录保留，可随时直接运行旧命令回退。更新包自带新的应用私有 Gdev／Runtime 和程序，不修改系统驱动或其他应用的 CUDA 路径。

更新使用 GK20A 的 coherent 缓冲映射；本机 `6.12.111-moze.1-native-vde2` 已验证。其他移植系统需自行确认 Nouveau 对 ARM coherent/WC 映射的支持，不能仅依据 GPU 型号判断兼容性。

诊断时可用 `MOCHA_LLM_FULL_LOGITS=1` 禁用候选压缩回传；`MOCHA_LLM_VERIFY_CANDIDATES=1` 会逐步对照完整词表的 Top-k，适用于 `cuda` 且 `top-k <= 64`。`--check` 总是读取全部 logits 并与 CPU 比较。正常推理的 Top-k 大于 64 时自动使用完整回传，采样参数保持原义。

## 输出长度与上下文

v0.2.1 的 Qwen 总上下文为 1024 token，包含提示词、ChatML 模板和输出。`-n 256` 可以正常使用，例如：

```sh
./mocha-llm -i '用中文介绍Linux。' -n 256
```

模型可能提前输出结束标记；这不是报错，也不保证恰好输出 256 个汉字。若提示词很长，程序会显示剩余输出预算，应缩短提示词或按提示降低 `-n`。TinyStories 的 512 步总长度含义保持原样。

Qwen 的 GPU KV 缓冲从 6 MiB 增到 24 MiB；纯 CUDA 不再额外预分配 CPU KV，CPU／hybrid／数值对照才分配主机 KV。模型权重与分词器没有变化。长上下文的注意力计算会更慢，已有的短提示词速度不代表满 1024 token 时的速度。
## v0.3.0：大模型

v0.3.0 更新包在原有两个模型之外，支持 Llama 3.2 1B Instruct 与 Qwen2.5 1.5B Instruct。新增模型均为 GGUF Q4_K_M，通过独立的 llama.cpp ARMv7 NEON CPU 引擎运行。原有 `--model qwen --backend cuda` 保留。

```sh
# 进入现有安装的父目录；旧版本留作回退
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.3.0/mocha-llm-v0.3.0-armhf-update.tar.gz
curl -fLo SHA256SUMS-v0.3.0 https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.3.0/SHA256SUMS
sha256sum -c SHA256SUMS-v0.3.0
tar -xzf mocha-llm-v0.3.0-armhf-update.tar.gz
cd mocha-llm-v0.3.0
python3 tools/reuse-models.py --from ../mocha-llm-v0.2.1
python3 tools/fetch-models.py --model 1b
python3 tools/fetch-models.py --model 1.5b
./mocha-llm --list-models
./mocha-llm --model 1b -i 'Explain Linux in one sentence.' -n 64 -t 0
./mocha-llm --model 1.5b -i '请用一句话介绍Linux。' -n 64 -t 0
```

也可以从 v0.1.0 基础包或 v0.2.0 安装目录复用原有权重，大小和哈希必须完全匹配。`verify-install.py` 校验更新包及旧模型；可选 GGUF 权重由 `fetch-models.py` 独立校验。

### 空间与内存

1B 文件为 807,694,464 字节，1.5B 为 1,117,320,736 字节，总计约 1.79 GiB。不要把它们装到空间不足的 APP 分区。下载器要求每个目标分区额外保留 256 MiB，空间不足时会在下载前退出。

```sh
# 先选有足够空间且当前账户可写的目录
python3 tools/fetch-models.py --model 1b --directory /path/to/first-disk/models
python3 tools/fetch-models.py --model 1.5b --directory /path/to/second-disk/models
# 可选镜像；来源修订和哈希保持不变
python3 tools/fetch-models.py --model 1.5b --base-url https://hf-mirror.com
```

工具在发行目录的 `models/` 中建立链接，因此更换模型无需重复修改启动命令。断开的下载保留 `.partial`，再次执行可续传。已有文件仍会重新校验；不覆盖损坏文件，也不关闭 TLS 校验。

建议先退出浏览器和不用的应用，暂停动态壁纸。启动前分别要求至少 1050 MiB（1B）、1150 MiB（1.5B）MemAvailable；这是避免明显内存不足的门槛，不是最大 RSS 保证。量化文件可通过 mmap 使用文件缓存，仍需 KV cache、计算缓冲和桌面内存。zram 不能替代足够的物理内存。

默认三线程、1024 总上下文、32-token 微批次、不使用 GPU offload、不锁定内存。`-n` 表示最多生成的新 token，建议先用 64／96；长提示词和输出共享总上下文，GGUF 引擎关闭自动上下文滑动。一次运行只加载选中的模型；切换前退出上一次推理。

1B 的主要支持语言不含中文，中文问答优先选 1.5B。Llama 具有独立的 Meta 许可证和使用政策，见 `licenses/`；工具代码的 MIT 不会改变模型许可。

卸载可选模型时，先退出推理，再删除其 `models/` 链接和实际外部权重文件；保留其他模型。回退只需运行旧版本目录中的 `mocha-llm`。更新不更改系统 GPU 驱动或全局运行库。

## v0.4.0：可选 Q4 CUDA

下载并校验 v0.4.0 更新包，解压到旧版本旁边。旧权重不需要重新下载：

```sh
curl -fLO https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.4.0/mocha-llm-v0.4.0-armhf-update.tar.gz
curl -fLo SHA256SUMS-v0.4.0 https://github.com/Pisces-Moze/mocha-moze-llm/releases/download/v0.4.0/SHA256SUMS
sha256sum -c SHA256SUMS-v0.4.0
tar -xzf mocha-llm-v0.4.0-armhf-update.tar.gz
cd mocha-llm-v0.4.0
python3 tools/reuse-models.py --from ../mocha-llm-v0.3.0
python3 tools/fetch-models.py --model 1b --directory ../mocha-llm-v0.3.0/models
python3 tools/fetch-models.py --model 1.5b --directory ../mocha-llm-v0.3.0/models
./mocha-llm --model 1.5b --backend cuda --gpu-cache-mib 64 -i '你好' -n 16
```

最后两个复用命令会核对完整哈希；若旧版本没有安装对应模型，则按清单下载。旧权重若在外部目录，更新后的链接直接指向实际文件，避免依赖旧版本的中间链接。若实际文件仍位于旧版本内部，则卸载旧版本前仍须保留或迁移它。

大模型默认仍为 CPU。CUDA 桥接只接管支持的量化矩阵，不等于完整 PyTorch／AirLLM，也不保证加速。流式上传和算子调度的代价、内存开销及已撤回的大缓存实验见 [说明](streamed-cuda.md)。
