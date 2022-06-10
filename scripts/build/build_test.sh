# Build only the testing executable

mkdir -p build
cd build
cmake ..
cmake --build . -j8 -t test_main
cd ..
