# Run the tests as well as generate code coverage

#!/bin/sh
# gcno files are built when object is compiled
# gcda files are built after execution
TMP_SPDLOG_LEVEL=${SPDLOG_LEVEL}
unset SPDLOG_LEVEL
find build/src -name "*.gcda" -type f -delete
printf "\nRunning CUDA Tests:\n"
./build/bin/test_main_cuda
printf "\nRunning CPU Tests:\n"
./build/bin/test_main_cpu
printf "\nRunning Lcov..."
lcov --directory . --capture -q --output-file build/code_coverage.info --exclude "*/usr/**/*" --exclude "*_deps/**/*" --exclude "*main.cpp" --exclude "/tmp/*" --exclude "*.cuh" --exclude "*/external/*"
printf "\nRunning genhtml..."
genhtml -q build/code_coverage.info --output-directory ./docs/code_coverage/
printf "\n"
find test_objects/tmp/ -not -name  "*.gitignore" -type f -delete
SPDLOG_LEVEL=${TMP_SPDLOG_LEVEL}
unset TMP_SPDLOG_LEVEL