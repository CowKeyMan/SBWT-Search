# Build the main executable, tests as well as documentation

mkdir -p build
cd build
cmake ..
cmake --build . -j8
cp compile_commands.json ..
cd ../documentation
doxygen Doxyfile
make html
cd ..
cp -R documentation/build/html/ docs/documentation
