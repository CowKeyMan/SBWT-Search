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

# populate combined list
combined_input_file=test_objects/full_pipeline/index_search/combined_input.list
combined_output_file=test_objects/full_pipeline/index_search/combined_output.list
printf "" > ${combined_input_file}
printf "" > ${combined_output_file}
input_files=(`cd test_objects/full_pipeline/index_search/ && find *.fna *.fnq`)
for file in ${input_files[@]}; do
  echo test_objects/full_pipeline/index_search/${file} >> ${combined_input_file}
  echo test_objects/full_pipeline/index_search/actual/${file%.*} >> ${combined_output_file}
done

for streams in {1..5}; do
  echo "Running combined with streams = ${streams}"
  for mode in ${modes[@]}; do
    ./build/bin/sbwt_search index \
      -o test_objects/full_pipeline/index_search/combined_output.list \
      -i test_objects/search_test_index.sbwt \
      -q test_objects/full_pipeline/index_search/combined_input.list \
      -p ${mode} \
      -s ${streams} \
      -c 0.1
  done
  run_tests
done

echo "Running individually"
for mode in ${modes[@]}; do
  for file in ${input_files[@]}; do
    ./build/bin/sbwt_search index \
      -q "test_objects/full_pipeline/index_search/${file}" \
      -i test_objects/search_test_index.sbwt \
      -o "test_objects/full_pipeline/index_search/actual/${file%.*}"  \
      -p ${mode} \
      -s 1 \
      -c 0.1
  done
done
run_tests

rm test_objects/full_pipeline/index_search/combined*.list

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
else
  echo "Index pipeline test passed!"
fi
