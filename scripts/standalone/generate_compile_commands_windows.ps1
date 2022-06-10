# Some developers may want to work on windows and use a compilation database
# for their ide or language server. This script creates the compilation database
# and copies it to the main folder
# The build occurs in a separate folder than the linux build

mkdir -p build_windows
cd build_windows
cmake -G "NMake Makefiles" .. -DENABLE_CLANG_TIDY=OFF -DENABLE_HEADER_GUARDS_CHECK=OFF
cp compile_commands.json ..
cd ..
