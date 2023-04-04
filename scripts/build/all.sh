#!/bin/bash

# Build the compilation commands, main executable in release mode, tests as
# well as documentation. It takes a single argument which is one of NVIDIA, AMD
# or CPU.

if [ $# -ne 1 ] || ( [ "${1,,}" != "nvidia" ] && [ "${1,,}" != "amd" ] && [ "${1,,}" != "cpu" ]); then
  echo "Usage: ./scripts/build/all.sh <NVIDIA|AMD|CPU|[other]>"
  exit 1
fi

./scripts/build/compilation_commands.sh "$1"

mkdir -p build
cd build

./scripts/build/release.sh "$1"
if [ $? -ne 0 ]; then >&2echo "Building release failed" && cd .. && exit 1; fi
./scripts/build/docs.sh
if [ $? -ne 0 ]; then >&2echo "Building docs failed" && cd .. && exit 1; fi
./scripts/build/tests.sh "$1"
if [ $? -ne 0 ]; then >&2echo "Building tests failed" && cd .. && exit 1; fi
