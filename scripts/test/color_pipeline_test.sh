#!/bin/bash

# Build the main executable, run the executable on
# test_objects/full_pipeline/color_search/ and verify that the outputs are
# correct (ie equal to the expected values, which can be found in the same
# folder)


cd test_objects/full_pipeline/color_search/expected
files=`ls *.txt`
cd ../../../..

modes=(
  "ascii"
  # "binary"
  # "csv"
)
extensions=(
  ".txt"
  # ".bin"
  # ".csv"
)

bad_exits=0

function run_tests() {
  for file in ${files}
  do
    no_extension="${file%%.*}"
    for extension in ${extensions[@]}
    do
      actual="${no_extension}${extension}"
      python3 scripts/test/verify_color_results_equal.py \
        -x "test_objects/full_pipeline/color_search/expected/${file}" \
        -y "test_objects/full_pipeline/color_search/actual/${actual}" \
        --quiet
      last_exit=$?
      if [ ${last_exit} -ne 0 ]; then
        echo ${file} and ${actual} do not match
        echo ""
      fi
      ((bad_exits+=${last_exit}))
    done
  done

  rm test_objects/full_pipeline/color_search/actual/*
}


# build
./scripts/build/release.sh

for streams in {1..5}; do
  echo "Running combined with streams = ${streams}"
  for mode in ${modes[@]}; do
    ./build/bin/sbwt_search colors -o test_objects/full_pipeline/color_search/combined_output.list -i test_objects/themisto_example/GCA_combined_d1.tcolors -q test_objects/full_pipeline/color_search/combined_input.list -s ${streams} -c ${mode} -t 0.7 -p 0.1
  done
  run_tests
done

echo "Running individually"
input_files=(`cat test_objects/full_pipeline/color_search/combined_input.list`)
output_files=(`cat test_objects/full_pipeline/color_search/combined_output.list`)
for mode in ${modes[@]}; do
  for i in ${!input_files[@]}; do
    ./build/bin/sbwt_search colors -o "${output_files[i]}" -i test_objects/themisto_example/GCA_combined_d1.tcolors -q "${input_files[i]}" -s 1 -c ${mode} -t 0.7 -p 0.1
  done
done
run_tests

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
else
  echo "Color pipeline test passed!"
fi
