# Build the main executable, tests as well as documentation

mkdir -p build
cd build
cmake ..
cmake --build . -j8
cd ../docs
doxygen Doxyfile
make html
cd ..
