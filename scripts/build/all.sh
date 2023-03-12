#!/bin/bash

# Build the main executable, tests as well as documentation in debug mode.
# If any argument at all is passed to this script, it will skip the cmake step and just execute the build step only

./scripts/build/compilation_commands.sh

mkdir -p build
cd build

# ./scripts/build/debug.sh
# ./scripts/build/release_nvidia.sh
./scripts/build/release_cpu.sh
if [ $? -ne 0 ]; then >&2echo "Building release failed" && cd .. && exit 1; fi
# ./scripts/build/release_amd.sh
./scripts/build/docs.sh
if [ $? -ne 0 ]; then >&2echo "Building docs failed" && cd .. && exit 1; fi
./scripts/build/tests.sh
if [ $? -ne 0 ]; then >&2echo "Building tests failed" && cd .. && exit 1; fi
