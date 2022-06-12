# Build only the documentation

mkdir -p build
cd build
cmake ..
cmake --build . -j8 -t docs
cd ../documentation
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/. docs/documentation
