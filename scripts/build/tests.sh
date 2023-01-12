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
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_HEADER_GUARDS_CHECK=OFF \
    -DENABLE_CLANG_FORMAT_CHECK=OFF \
    -DBUILD_VERIFY=OFF \
    -DBUILD_MAIN=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=ON \
    -DENABLE_MARCH_NATIVE=OFF \
    ..
fi
cmake --build . -j8
cd ..
