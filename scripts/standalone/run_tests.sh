# Run the tests as well as generate code coverage

#!/bin/sh
echo "\nRunning Tests:"
./build/src/test_main
echo "\nRunning Lcov:"
lcov --directory . --capture --output-file build/code_coverage.info --exclude "*/usr/**/*" --exclude "*_deps/**/*" --exclude "*/test_main.cpp"
echo "\nRunning genhtml:"
genhtml build/code_coverage.info --output-directory ./docs/code_coverage/
