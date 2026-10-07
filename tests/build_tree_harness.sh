#!/bin/sh
# Build the bit-exactness harness against a llama.cpp tree that carries the KVarN cache (read-only use of the tree).
# usage: tests/build_tree_harness.sh <tree root> <tree build dir with bin/libggml-base.so and bin/libggml-cpu.so>
set -e
TREE=${1:?tree root}
BUILD=${2:?tree build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function \
    -I"$TREE/ggml/include" -I"$TREE/ggml/src" -I"$HERE" \
    "$HERE/tree_harness.cpp" -o "$HERE/tree_harness" \
    -L"$BUILD/bin" -lggml-base -lggml-cpu -Wl,-rpath,"$BUILD/bin" -lm -lpthread
echo "built $HERE/tree_harness"
