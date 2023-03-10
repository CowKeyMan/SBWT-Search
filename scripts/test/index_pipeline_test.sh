#!/bin/bash

# Build the main executable, run the executable on
# test_objects/full_pipeline/index_search/ and verify that the outputs are
# correct (ie equal to the expected values, which can be found in the same folder)

# build
./scripts/build/release.sh
# run
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 5 -c ascii
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 4 -c binary
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 4 -c bool

cd test_objects/full_pipeline/index_search/expected
files=`ls *.txt`
cd -

extensions=(
  ".txt"
  ".bin"
  ".bool"
)

bad_exits=0

for file in ${files}
do
  no_extension="${file%%.*}"
  for extension in ${extensions[@]}
  do
    actual="${no_extension}${extension}"
    echo Checking differences between ${file} and ${actual}
    python3 scripts/test/verify_index_results_equal.py \
      -x "test_objects/full_pipeline/index_search/expected/${file}" \
      -y "test_objects/full_pipeline/index_search/actual/${actual}"
    ((bad_exits+=$?))
  done
done

rm test_objects/full_pipeline/index_search/actual/*

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
fi
