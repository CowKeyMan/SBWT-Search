# Build only the documentation

mkdir -p build
cd build
cmake -DENABLE_CLANG_TIDY=OFF -DENABLE_HEADER_GUARDS_CHECK=OFF -DENABLE_CLANG_FORMAT_CHECK=OFF ..
cmake --build . -j8 -t docs
cd ../documentation
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/. docs/documentation
