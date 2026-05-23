#!/bin/bash
# Build script for LubanCat 2 (RK3568 aarch64)
set -e

BUILD_DIR=build_lubancat2
TOOLCHAIN=aarch64-linux-gnu-

export CC=${TOOLCHAIN}gcc
export CXX=${TOOLCHAIN}g++
export STRIP=${TOOLCHAIN}strip

cmake -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DONNXRUNTIME_DIR=/usr/lib/aarch64-linux-gnu \
    -DRKNNLLM_DIR=/usr/lib

cmake --build $BUILD_DIR -j4
$STRIP $BUILD_DIR/rag_cli

echo "Done. Binary: $BUILD_DIR/rag_cli"
echo "Copy to LubanCat 2: scp $BUILD_DIR/rag_cli cat@lubancat2:/home/cat/rag/"
