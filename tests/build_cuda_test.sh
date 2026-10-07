#!/bin/sh
# Build the CUDA reference-kernel test. usage: tests/build_cuda_test.sh [sm arch, default 86]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
nvcc -std=c++17 -O2 -arch=sm_${1:-86} -I"$HERE" "$HERE/test_cuda.cu" -o "$HERE/test_cuda"
echo "built $HERE/test_cuda"
