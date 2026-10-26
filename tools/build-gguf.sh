#!/bin/sh
# SPDX-License-Identifier: MIT
# Build the pinned upstream CPU engine, not a CUDA GGUF backend.
set -eu
if [ "$#" -ne 2 ]; then echo "usage: $0 LLAMA_CPP_SOURCE OUTPUT_DIRECTORY" >&2; exit 2; fi
source_dir=$(realpath "$1"); mkdir -p "$2"; output=$(realpath "$2")
expected=ec7f3ac9ab33e46b136eb5ab6a76c4d81f57c7f1
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected"
python3 "$(dirname "$0")/patch-gguf-cuda.py" "$source_dir"
cat > "$output/armv7.cmake" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_C_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee")
set(CMAKE_CXX_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee")
EOF
cmake -S "$source_dir" -B "$output/build" -DCMAKE_TOOLCHAIN_FILE="$output/armv7.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=ON -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$output/build" -j2 --target llama-cli
cp "$output/build/bin/llama-cli" "$output/mocha-gguf-engine"
