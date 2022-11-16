# Build the main executable

mkdir -p build
cd build
if [[ $# > 0 ]];
then
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_HEADER_GUARDS_CHECK=OFF \
    -DENABLE_CLANG_FORMAT_CHECK=OFF \
    -DBUILD_CPU=ON \
    -DBUILD_CUDA=ON \
    -DBUILD_MAIN=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DENABLE_PROFILING=OFF \
    ..
fi
cmake --build . -j8
cd ..
