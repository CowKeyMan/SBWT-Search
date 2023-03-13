#!/bin/bash

# Build the main executable, run the executable on
# test_objects/full_pipeline/index_search/ and verify that the outputs are
# correct (ie equal to the expected values, which can be found in the same folder)

if [ "$1" != "nvidia" ] && [ "$1" != "cpu" ] && [ "$1" != "amd" ]; then
  echo "Invalid platform, must be nvidia, cpu or amd"
  exit 1
fi

# build
./scripts/build/release_$1.sh

cd test_objects/full_pipeline/index_search/expected
files=`ls *.txt`
cd -

modes=(
  ascii
  binary
)
extensions=(
  ".txt"
  ".bin"
)

bad_exits=0

function run_tests() {
  for file in ${files}
  do
    no_extension="${file%%.*}"
    for extension in ${extensions[@]}
    do
      actual="${no_extension}${extension}"
      python3 scripts/test/verify_index_results_equal.py \
        -x "test_objects/full_pipeline/index_search/expected/${file}" \
        -y "test_objects/full_pipeline/index_search/actual/${actual}" \
        --quiet
      last_exit=$?
      if [ ${last_exit} -ne 0 ]; then
        echo ${file} and ${actual} do not match
        echo ""
      fi
      ((bad_exits+=${last_exit}))
    done
  done

  rm test_objects/full_pipeline/index_search/actual/*
}

for streams in {1..5}; do
  echo "Running combined with streams = ${streams}"
  for mode in ${modes[@]}; do
    ./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -s ${streams} -p ${mode} -c 0.1
  done
  run_tests
done

echo "Running individually"
input_files=(`cat test_objects/full_pipeline/index_search/combined_input.list`)
output_files=(`cat test_objects/full_pipeline/index_search/combined_output.list`)
for mode in ${modes[@]}; do
  for i in ${!input_files[@]}; do
    ./build/bin/sbwt_search index -o "${output_files[i]}" -i test_objects/search_test_index.sbwt -q "${input_files[i]}"  -s 1 -p ${mode} -c 0.1
  done
done
run_tests

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
else
  echo "Index pipeline test passed!"
fi
