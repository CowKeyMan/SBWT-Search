# Build the main executable, tests as well as documentation in debug mode

mkdir -p build
cd build
cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CLANG_TIDY=ON \
  -DENABLE_HEADER_GUARDS_CHECK=ON \
  -DENABLE_CLANG_FORMAT_CHECK=ON \
  -DBUILD_CPU=ON \
  -DBUILD_CUDA=ON \
  -DBUILD_MAIN=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  ..
cp compile_commands.json ..
cd ..


sh scripts/build/build_release.sh
sh scripts/build/build_benchmarks.sh
sh scripts/build/build_tests.sh
sh scripts/build/build_docs.sh
