# 部署教程

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

日常交互先试三线程 CPU，实测对比见性能记录。CUDA 用于原生 GPU 推理和移植验证。两个后端均使用量化模型，CPU 后端的矩阵乘包含 NEON 优化。线程设为 4 可能抢占桌面资源，先比较 2 与 3。

如需比较 CUDA 的线程块，可在单次命令前设置：

```sh
MOCHA_LLM_CUDA_THREADS=1024 ./mocha-llm --backend cuda -i '你好' -n 32
```

Qwen 默认 1024，也支持 512；实测 1024 稍快。此参数不会修改 GPU 频率。不要把不同温度、提示词长度或 `--check` 状态的速度直接比较。

## 6. 卸载和恢复

退出命令会释放 GPU 权重和 KV cache。没有常驻服务、系统启动项或全局图形库替换。

如果按 README 解压到 `~/Applications/mocha-llm`，删除自己解压出的 `mocha-llm-v0.1.0` 文件夹即可卸载该版本。若手动建立过命令软链接，再删除对应软链接。已有系统 CUDA 实验目录和旧模型不受影响。
