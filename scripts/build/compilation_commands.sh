#!/bin/bash

# Generate the compilation commands database (in build/compile_commands.json)

mkdir -p build
cd build
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_MAIN=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_DOCS=OFF \
  -DENABLE_PROFILING=ON \
  -DENABLE_MARCH_NATIVE=OFF \
  -DHIP_TARGET_DEVICE=CPU \
  -DROCM_BRANCH="rocm-5.4.x" \
  ..
if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
cd ..
