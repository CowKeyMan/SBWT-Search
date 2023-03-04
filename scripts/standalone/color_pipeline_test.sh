#!/bin/bash

# Build the main executable, run the executable on
# test_objects/full_pipeline/color_search/ and verify that the outputs are
# correct (ie equal to the expected values, which can be found in the same
# folder)

# build
# ./scripts/build/release.sh
# run
# ./build/bin/sbwt_search colors -o test_objects/full_pipeline/color_search/combined_output.list -i test_objects/themisto_example/GCA_combined_d1.tcolors -q test_objects/full_pipeline/color_search/combined_input.list -b 1 -c ascii -t 0.7

cd test_objects/full_pipeline/color_search/expected
files=`ls *.txt`
cd -

extensions=(
  ".txt"
  # ".bin"
  # ".csv"
)

bad_exits=0

for file in ${files}
do
  no_extension="${file%%.*}"
  for extension in ${extensions[@]}
  do
    actual="${no_extension}${extension}"
    echo Checking differences between ${file} and ${actual}
    python3 scripts/standalone/verify_color_results_equal.py \
      -x "test_objects/full_pipeline/color_search/expected/${file}" \
      -y "test_objects/full_pipeline/color_search/actual/${actual}"
    ((bad_exits+=$?))
  done
done

# rm test_objects/full_pipeline/color_search/actual/*

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
fi
