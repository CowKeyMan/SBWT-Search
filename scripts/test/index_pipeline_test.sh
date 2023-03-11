#!/bin/bash

# Build the main executable, run the executable on
# test_objects/full_pipeline/index_search/ and verify that the outputs are
# correct (ie equal to the expected values, which can be found in the same folder)

# build
./scripts/build/release.sh

cd test_objects/full_pipeline/index_search/expected
files=`ls *.txt`
cd -

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
      echo Checking differences between ${file} and ${actual}
      python3 scripts/test/verify_index_results_equal.py \
        -x "test_objects/full_pipeline/index_search/expected/${file}" \
        -y "test_objects/full_pipeline/index_search/actual/${actual}"
      ((bad_exits+=$?))
    done
  done

  rm test_objects/full_pipeline/index_search/actual/*

  echo ""
}

echo "Running combined with batches = 1"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 1 -c binary -p 0.1
run_tests

echo "Running combined with batches = 2"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 2 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 2 -c binary -p 0.1
run_tests

echo "Running combined with batches = 3"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 3 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 3 -c binary -p 0.1
run_tests

echo "Running combined with batches = 4"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 4 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 4 -c binary -p 0.1
run_tests

echo "Running combined with batches = 5"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 5 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/combined_output.list -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/combined_input.list -b 5 -c binary -p 0.1
run_tests

echo "Running individually"
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta1 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta1.fna -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta1 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta1.fna -b 1 -c binary -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta2 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta2.fna -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta2 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta2.fna -b 1 -c binary -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta3 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta3.fna -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta3 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta3.fna -b 1 -c binary -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta4 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta4.fna -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fasta4 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fasta4.fna -b 1 -c binary -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fastq1 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fastq1.fnq -b 1 -c ascii -p 0.1
./build/bin/sbwt_search index -o test_objects/full_pipeline/index_search/actual/fastq1 -i test_objects/search_test_index.sbwt -q test_objects/full_pipeline/index_search/fastq1.fnq -b 1 -c binary -p 0.1
run_tests

if [[ ${bad_exits} -gt 0 ]]; then
  exit 1
fi
