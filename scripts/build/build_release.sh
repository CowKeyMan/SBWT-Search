# Build the main executable, tests as well as documentation

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j8 -t main
cd ..
