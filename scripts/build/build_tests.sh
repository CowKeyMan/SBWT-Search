# Build the tests only

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_CUDA=ON -DBUILD_CPU=ON -DBUILD_MAIN=OFF -DBUILD_TESTS=ON .. # to build everything
cmake --build . -j8
cd ..
