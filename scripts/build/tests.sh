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
    -DHIP_TARGET_DEVICE=CPU \
    -DROCM_BRANCH="rocm-5.4.x" \
    ..
  if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
fi
cmake --build . -j8
if [ $? -ne 0 ]; then >&2 echo "Build" && cd .. && exit 1; fi
cd ..
