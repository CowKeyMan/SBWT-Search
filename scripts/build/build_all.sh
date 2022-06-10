# Build the main executable, tests as well as documentation in debug mode

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j8
cp compile_commands.json ..
cd ../documentation
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/ docs/documentation
