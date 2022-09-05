# Build the main executable, tests as well as documentation in debug mode

mkdir -p build
cd build
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CLANG_TIDY=OFF \
  -DENABLE_HEADER_GUARDS_CHECK=OFF \
  -DENABLE_CLANG_FORMAT_CHECK=OFF \
  -DBUILD_CPU=ON \
  -DBUILD_CUDA=ON \
  -DBUILD_MAIN=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_DOCS=OFF \
  -DENABLE_PROFILING=ON \
  ..
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CLANG_TIDY=ON \
  -DENABLE_HEADER_GUARDS_CHECK=ON \
  -DENABLE_CLANG_FORMAT_CHECK=ON \
  -DBUILD_CPU=OFF \
  -DBUILD_CUDA=OFF \
  -DBUILD_MAIN=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_DOCS=OFF \
  -DENABLE_PROFILING=OFF \
  ..
cd ..


sh scripts/build/release.sh
# sh scripts/build/debug.sh
# sh scripts/build/tests.sh
sh scripts/build/docs.sh
