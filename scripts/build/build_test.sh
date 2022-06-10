# Build only the testing executable

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j8 -t test_main
cd ..
