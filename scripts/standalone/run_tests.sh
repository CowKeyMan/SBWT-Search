# Run the tests as well as generate code coverage

#!/bin/sh
cd build
echo "\nRunning Tests:"
./src/test_main
echo "\nRunning Lcov:"
lcov --directory . --capture --output-file ./code_coverage.info --exclude "*/usr/**/*" --exclude "*_deps/**/*"
echo "\nRunning genhtml:"
genhtml code_coverage.info --output-directory ./code_coverage_report/
cd ..
