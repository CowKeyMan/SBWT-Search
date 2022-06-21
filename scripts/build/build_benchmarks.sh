# Build the benchmarks

mkdir -p build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CLANG_TIDY=OFF \
  -DENABLE_HEADER_GUARDS_CHECK=OFF \
  -DENABLE_CLANG_FORMAT_CHECK=OFF \
  -DBUILD_CPU=ON \
  -DBUILD_CUDA=ON \
  -DBUILD_MAIN=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=ON \
  ..
cmake --build . -j8
cd ..
