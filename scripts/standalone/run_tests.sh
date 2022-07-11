# Run the tests as well as generate code coverage

#!/bin/sh
printf "\nRunning Tests:"
./build/bin/test_main_cuda
./build/bin/test_main_cpu
printf "\nRunning Lcov..."
lcov --directory . --capture -q --output-file build/code_coverage.info --exclude "*/usr/**/*" --exclude "*_deps/**/*" --exclude "*main.cpp" --exclude "/tmp/*" --exclude "*.cuh" --exclude "*/external/*"
printf "\nRunning genhtml..."
genhtml -q build/code_coverage.info --output-directory ./docs/code_coverage/
printf "\n"
find test_objects/tmp/ -not -name  "*.gitignore" -type f -delete
