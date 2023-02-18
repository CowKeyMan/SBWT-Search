#!/bin/bash

# Build the main test executable.
# If any argument at all is passed to this script, it will skip the cmake step and just execute the build step only

mkdir -p build
cd build
if [[ $# == 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_MAIN=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=ON \
    -DENABLE_MARCH_NATIVE=OFF \
    -DHIP_TARGET_DEVICE=NVIDIA \
    -DROCM_BRANCH="rocm-5.4.x" \
    ..
fi
cmake --build . -j8
cd ..
