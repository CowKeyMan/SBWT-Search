#!/bin/bash

# Build the main executable in debug mode.
# If any argument at all is passed to this script, it will skip the cmake step and just execute the build step only

mkdir -p build
cd build
if [[ $# == 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_VERIFY=OFF \
    -DBUILD_MAIN=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=ON \
    -DENABLE_MARCH_NATIVE=OFF \
    ..
fi
cmake --build . -j8
cd ..
