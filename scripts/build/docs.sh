#!/bin/bash

# Build only the documentation.
# If any argument at all is passed to this script, it will skip the cmake step and just execute the build step only

cp documentation/Documents/FileDocumentation.rst.in documentation/Documents/FileDocumentation.rst
./scripts/standalone/generate_file_documentation.py >> documentation/Documents/FileDocumentation.rst
if [[ $? -ne 0 ]]; then >&2 echo "Generating file documentation failed." && exit 1; fi

mkdir -p build
cd build
if [[ $# == 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_MAIN=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=ON \
    -DENABLE_PROFILING=OFF \
    -DENABLE_MARCH_NATIVE=OFF \
    ..
  if [ $? -ne 0 ]; then >&2echo "Cmake generation failed" && cd .. && exit 1; fi
fi
cmake --build . -j8
if [ $? -ne 0 ]; then >&2 echo "Build" && cd .. && exit 1; fi
cd ../documentation
rm `grep -r -L -P "(.gitignore|.dir_docstring)" RST`
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/. docs/documentation
