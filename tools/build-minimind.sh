#!/bin/sh
# CUDA 6.5 is used only to produce native sm_32 code for our open runtime.
set -eu
if [ "$#" -ne 4 ]; then echo "usage: $0 CUDA_ROOT OLD_TOOLCHAIN_ROOT RUNTIME_BUILD OUTPUT" >&2; exit 2; fi
cuda=$(realpath "$1"); old=$(realpath "$2"); runtime=$(realpath "$3")
mkdir -p "$4"; output=$(realpath "$4")
here=$(CDPATH= cd -- "$(dirname -- "$0")/../src" && pwd)
defines=${MOCHA_MODEL_DEFINES:-}
arm-linux-gnueabihf-g++ $defines -std=c++11 -O3 -ffp-contract=off -fopenmp \
    -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard \
    -ffunction-sections -fdata-sections -c "$here/minimind.cpp" -o "$output/host.o"
inc="$old/usr/arm-linux-gnueabihf/include"
LD_LIBRARY_PATH="$old/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$cuda/bin/nvcc" $defines -target-cpu-arch ARM -m32 -O3 --fmad=false \
    -ccbin "$old/usr/bin/arm-linux-gnueabihf-g++-4.8" \
    -Xcompiler=-B/usr/arm-linux-gnueabihf/bin/,-ffunction-sections,-fdata-sections \
    -I"$inc/c++/4.8.2" -I"$inc/c++/4.8.2/arm-linux-gnueabihf" \
    -I"$inc/c++/4.8.2/backward" -I"$inc" \
    -I"$cuda/targets/armv7-linux-gnueabihf/include" \
    --cudart=none -arch=sm_32 -c "$here/minimind.cu" -o "$output/device.o"
arm-linux-gnueabihf-g++ -fopenmp -no-pie -Wl,--gc-sections "$output/host.o" "$output/device.o" \
    -L"$runtime/cuda" -L"$runtime/lib" -Wl,-rpath-link,"$runtime/lib" -lucuda -lm \
    -o "$output/mocha-minimind-engine"
