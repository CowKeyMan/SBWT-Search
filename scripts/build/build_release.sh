# Build the main executable, tests as well as documentation

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CLANG_TIDY=OFF -DENABLE_HEADER_GUARDS_CHECK=OFF -DENABLE_CLANG_FORMAT_CHECK=OFF ..
cmake --build . -j8 -t main
cd ..
