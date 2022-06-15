# Run the tests as well as generate code coverage

#!/bin/sh
echo "\nRunning Tests:"
./build/src/test_main_cuda
./build/src/test_main_cpu
echo "\nRunning Lcov:"
lcov --directory . --capture --output-file build/code_coverage.info --exclude "*/usr/**/*" --exclude "*_deps/**/*" --exclude "*main.cpp" --exclude "/tmp/*" --exclude "*.cuh"
echo "\nRunning genhtml:"
genhtml build/code_coverage.info --output-directory ./docs/code_coverage/
