# 构建与模型转换

预编译发行包是普通用户的安装入口。本页面向重建引擎、检查开放 Runtime 或修改 CUDA 内核的开发者。构建在 x86_64 Linux 上交叉完成，平板不需要开发工具链。

## 模型来源

Qwen 固定为 `Qwen/Qwen2.5-0.5B-Instruct` 的 `7ae557604adf67be50417f59c2c2f167def9a775`。下载该版本的 `model.safetensors`，源 SHA256 和配置见 `models/qwen-inputs.json`。

权重转换已使用 Python 3.14 和 NumPy 2.5.2；设备端不依赖 NumPy。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numpy==2.5.2 -i https://mirrors.aliyun.com/pypi/simple
python3 tools/convert-qwen.py /path/to/model.safetensors /path/to/qwen-q8.bin
```

转换前强制核对源权重 SHA256。BF16 按位展开到 FP32，再每 64 个连续权重对称量化为有符号 INT8；缩放系数、RMSNorm 和 QKV bias 保持 FP32。格式是本项目专用格式，不是 GGUF。模型权重和分词器必须配套。

故事模型输入由 `src/stories/fetch-llm-inputs.py` 按 `models/stories-inputs.json` 固定的 URL 和哈希获取。它使用 Karpathy 的原生二进制格式。

## 交叉工具链

已使用 GCC 14 的 `arm-linux-gnueabihf-g++` 编译 host，CUDA 6.5 NVCC 和隔离的 Ubuntu GCC 4.8 编译设备代码。NVCC 6.5 仍支持 Tegra 的 ARMv7 hard-float／`sm_32` 目标。CUDA 编译器及其头文件需自行从 NVIDIA 获取，本仓库和发行包不提供这些文件。

旧 GCC 前缀应包含 `usr/bin/arm-linux-gnueabihf-g++-4.8` 及对应 ARMhf 头文件。它只在 NVCC 那一步使用，不替换宿主机系统编译器。

开放 Runtime 的固定来源在 `runtime-source/sources.json`。构建机需要 Python 3.13+、CMake、patch、dpkg-deb、现代 ARMhf 交叉 GCC，以及 ARMhf Boost.Thread、libdrm 和 libdrm_nouveau 开发包。Debian 13 构建环境可以先安装：

```sh
sudo dpkg --add-architecture armhf
sudo apt-get update
sudo apt-get install python3 python3-venv cmake make patch curl dpkg-dev \
  gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf \
  libboost-thread-dev:armhf libdrm-dev:armhf
python3 tools/prepare-toolchain.py build/inputs
```

`prepare-toolchain.py` 根据清单下载并校验 NVIDIA 编译输入、隔离 GCC 4.8 的 Ubuntu 包和 Gdev 源码，只解包到指定目录。请先接受 NVIDIA 下载文件适用的许可；这些文件不随本项目再分发。旧 Ubuntu 包只供 NVCC 交叉编译使用，不能安装进 Debian 系统。中国大陆构建时可为当前 Debian 版本配置阿里云镜像，Python 包使用上面的阿里云 PyPI 地址。

Qwen 源权重也可用 `python3 tools/fetch-qwen.py build/qwen-inputs` 获取并校验。故事模型执行 `python3 src/stories/fetch-llm-inputs.py build/stories-inputs`。

`runtime-source/build-runtime.py` 对 Gdev 原始压缩包校验后应用 Mocha patch，再用 CMake 构建。

```sh
python3 runtime-source/build-runtime.py /path/to/gdev-master.tar.gz /path/to/new-runtime-build
```

此目录必须尚不存在。模型引擎通过 `libucuda` 链接开放实现，不链接 NVIDIA 专有 libcudart。

下载器输出的 CUDA 前缀是 `build/inputs/toolkit/usr/local/cuda-6.5`，旧 GCC 前缀是 `build/inputs/old-toolchain-root`，Gdev 归档在 `build/inputs/downloads/gdev-master.tar.gz`。以下命令中的前缀可替换为这些路径。

```sh
MOCHA_MODEL_DEFINES=-DMOCHA_QWEN sh tools/build-minimind.sh \
  /path/to/cuda-6.5 /path/to/old-toolchain-root \
  /path/to/new-runtime-build/build /path/to/qwen-output

sh src/stories/build-llm.sh /path/to/stories-inputs \
  /path/to/cuda-6.5 /path/to/old-toolchain-root \
  /path/to/new-runtime-build/build /path/to/stories-output
```

第一个脚本的输出名为历史名称 `mocha-minimind-engine`，打包时重命名为 `mocha-qwen-engine`。`MOCHA_QWEN` 选择 Qwen 形状、bias、RMS epsilon 和词表；不能拿 MiniMind 构建去读取 Qwen 权重。

## 发布包

先将两个引擎和转换后的权重在设备上验证，再执行：

```sh
python3 tools/package.py \
  --qwen-engine /path/to/qwen-output/mocha-minimind-engine \
  --stories-engine /path/to/stories-output/mocha-cuda-llm-engine \
  --gguf-engine /path/to/gguf-output/mocha-gguf-engine \
  --qwen-weights /path/to/qwen-q8.bin \
  --stories-inputs /path/to/stories-inputs \
  --runtime-prefix /path/to/tested-runtime-prefix \
  --output /path/to/new-release-directory
```

打包器只复制明确列出的开放库、程序和模型，生成每个文件的 SHA256 清单，以及压缩包校验文件。权重随 Release 分发；Git 只保存实现、分词器、来源、许可证和实测记录。

当前打包器默认版本为 `0.3.0`。增加 `--reuse-models` 可生成不含原有三份模型输入的更新包；清单仍包含这些文件的预期大小与哈希，接收端必须运行 `tools/reuse-models.py --from OLD_RELEASE` 后再校验。构建和测试所用的 Runtime 前缀需同时包含新 `libgdev.so.1.0.0` 与新 `libucuda.so.1.0.0`，只替换后者无法得到 coherent 缓冲优化。

常规测试：`python3 -m unittest discover -s tests -v`。真实 CUDA 计算、数值与性能结果需要在 GK20A 上重新取得；CI 不声称验证 GPU。

## 大缓冲区回归

Runtime 修复了旧 Gdev 的 generic shared/local 地址窗口与大模型权重冲突。独立回归分配 512 MiB，写入确定性图案，再由 GPU 使用 generic load 读取每个 4 KiB 页的样本；仅在 CPU 映射上读回无法覆盖这个问题。

```sh
sh tools/build-memory-probe.sh CUDA_ROOT OLD_TOOLCHAIN_ROOT RUNTIME_BUILD build/memory-probe
# 把程序放到平板发行包目录；至少保留 700 MiB 可用内存。
LD_LIBRARY_PATH="$PWD/runtime/lib" ./mocha-large-buffer-probe
```

应输出 `PAGE_SAMPLES=131072 ERRORS=0`。这是针对窗口冲突的回归，不代表每个字节都经过 GPU 全量校验。随后用 `./mocha-llm --backend cuda --check -i '你好' -n 8 -t 0` 覆盖真实 Transformer 路径。
## v0.3.0 GGUF 引擎

两款大模型复用 llama.cpp b4514，固定提交 `ec7f3ac9ab33e46b136eb5ab6a76c4d81f57c7f1`，不将其全部源码复制进仓库。

```sh
git clone https://github.com/ggml-org/llama.cpp.git build/llama.cpp
git -C build/llama.cpp checkout ec7f3ac9ab33e46b136eb5ab6a76c4d81f57c7f1
sh tools/build-gguf.sh build/llama.cpp build/gguf-armhf
```

使用现有现代 ARMhf 交叉编译器，指定 ARMv7、NEON VFPv4、hard-float 和 IEEE FP16 存储格式；静态链接 ggml／llama 组件，保留系统 glibc、libstdc++、libgomp 等依赖。禁用 CUDA、联网下载、服务器和测试构建。该 CPU 引擎与专用 Q8 CUDA 引擎独立。

打包器默认版本为 `0.3.0`，需额外传入 `--gguf-engine build/gguf-armhf/mocha-gguf-engine`。其余参数与原有更新包一致。GGUF 权重不进入更新包；来源和校验值在 `models/large-models.json`，部署时按需下载。
