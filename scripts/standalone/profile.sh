#!/bin/bash

# First build the release build

mkdir -p build
cd build
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MAIN=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DOCS=OFF \
  -DENABLE_PROFILING=ON \
  ..
cmake --build . -j8
cd ..
gprof
