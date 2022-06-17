# Build only the documentation

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CLANG_TIDY=OFF -DENABLE_HEADER_GUARDS_CHECK=OFF -DENABLE_CLANG_FORMAT_CHECK=OFF -DBUILD_CPU=OFF -DBUILD_CUDA=OFF -DBUILD_MAIN=OFF -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF .. # to build nothing
cd ../documentation
rm `grep -r -L -P "(.gitignore)" RST`
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/. docs/documentation
