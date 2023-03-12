#!/bin/bash

# Build the main executable for the AMC/ROCm platform. If any argument at all
# is passed to this script, it will skip the cmake step and just execute the
# build step only.

mkdir -p build
cd build
if [[ $# == 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MAIN=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=OFF \
    -DENABLE_MARCH_NATIVE=OFF \
    -DHIP_TARGET_DEVICE=AMD \
    -DROCM_BRANCH="rocm-5.4.x" \
    ..
  if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
fi
cmake --build . -j8
if [ $? -ne 0 ]; then >&2 echo "Build" && cd .. && exit 1; fi
cd ..
