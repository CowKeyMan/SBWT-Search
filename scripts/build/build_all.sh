# Build the main executable, tests as well as documentation in debug mode

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CLANG_TIDY=ON -DENABLE_HEADER_GUARDS_CHECK=ON -DENABLE_CLANG_FORMAT_CHECK=ON -DBUILD_CPU=ON -DBUILD_CUDA=ON -DBUILD_MAIN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON .. # to build everything
cmake --build . -j8
cp compile_commands.json ..
cd ../documentation
rm `grep -r -L -P "(gitignore)" RST`
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/. docs/documentation
