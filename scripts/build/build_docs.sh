# Build only the documentation

mkdir -p build
cd build
cmake ..
cmake --build . -j8 -t docs
cd ../docs
doxygen Doxyfile
make html
cd ..
