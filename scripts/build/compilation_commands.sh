#!/bin/bash

# Generate the compilation commands database for the target platform. It takes
# a single argument which is one of NVIDIA, AMD or CPU. The result will be
# found in build/compile_commands.json.

if [ $# -ne 1 ] || ( [ "${1,,}" != "nvidia" ] && [ "${1,,}" != "amd" ] && [ "${1,,}" != "cpu" ]); then
  echo "Usage: ./scripts/build/compilation_commands.sh <NVIDIA|AMD|CPU>"
  exit 1
fi

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
  -DHIP_TARGET_DEVICE="$1" \
  -DROCM_BRANCH="rocm-5.4.x" \
  ..
if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
cd ..
