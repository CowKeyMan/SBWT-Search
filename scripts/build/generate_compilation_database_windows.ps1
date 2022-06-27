mkdir build_windows -ErrorAction SilentlyContinue
cd build_windows
cmake `
  -G "NMake Makefiles" `
  .. `
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
  -DCMAKE_BUILD_TYPE=Debug `
  -DENABLE_CLANG_TIDY=OFF `
  -DENABLE_HEADER_GUARDS_CHECK=OFF `
  -DENABLE_CLANG_FORMAT_CHECK=OFF `
  -DBUILD_DOCS=OFF `
  -DBUILD_CPU=ON `
  -DBUILD_CUDA=ON `
  -DBUILD_MAIN=ON `
  -DBUILD_TESTS=ON `
  -DBUILD_BENCHMARKS=ON
cd ..