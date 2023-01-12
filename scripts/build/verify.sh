#!/bin/bash

# Build the verification executable to verify that 2 files contain the same content
# If any argument at all is passed to this script, it will skip the cmake step and just execute the build step only

mkdir -p build
cd build
if [[ $# == 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_HEADER_GUARDS_CHECK=OFF \
    -DENABLE_CLANG_FORMAT_CHECK=OFF \
    -DBUILD_VERIFY=ON \
    -DBUILD_MAIN=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=OFF \
    -DENABLE_MARCH_NATIVE=OFF \
    ..
fi
cmake --build . -j8
cd ..
