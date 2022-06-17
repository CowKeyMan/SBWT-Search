# Build the main executable

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CLANG_TIDY=OFF -DENABLE_HEADER_GUARDS_CHECK=OFF -DENABLE_CLANG_FORMAT_CHECK=OFF -DBUILD_CPU=ON -DBUILD_CUDA=ON -DBUILD_MAIN=OFF -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF .. # to build tests only
cmake --build . -j8
cd ..