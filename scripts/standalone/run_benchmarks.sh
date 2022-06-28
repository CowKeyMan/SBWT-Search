# Run the benchmark_main several times and save the results to a file


mkdir -p benchmark_results
mkdir -p tmp

DATETIME="$(date +"%Y-%m-%d_%I-%M-%S_%p")"
OUTPUT_FILE_NAME="benchmark_results/${DATETIME}_benchmark.txt"
TEMP_FILE_NAME="tmp/current_benchmark.txt"
BENCHMARK_AVERAGES_FILE_NAME="benchmark_results/averages.txt"

# Add or comment arguments from here
ARGUMENTS_LIST=(
  "-QUERY_FILE_PARSER_FASTA"
  "-QUERY_FILE_PARSER_FASTA_ZIP"
  "-QUERY_FILE_PARSER_FASTQ"
  "-QUERY_FILE_PARSER_FASTQ_ZIP"
  "-RAW_SEQUENCE_PARSER_CHAR_TO_BITS"
)
ARGUMENTS=""
for ARGUMENT in ${ARGUMENTS_LIST[@]};
do
  ARGUMENTS+="${ARGUMENT} "
done


# Run once so that we do not get any weird items taking longer than they should
build/bin/benchmark_main ${ARGUMENTS} > /dev/null 2>/dev/null
echo "Finished dummy benchmark run"
for i in {1..4}
do
  build/bin/benchmark_main ${ARGUMENTS} >> "${TEMP_FILE_NAME}" 2>/dev/null
  printf '\n' >> "${TEMP_FILE_NAME}"
  echo "Finished benchmark run ${i}"
done

cat "${TEMP_FILE_NAME}" > "${OUTPUT_FILE_NAME}"
rm "${TEMP_FILE_NAME}"

echo "${DATETIME}" >> "${BENCHMARK_AVERAGES_FILE_NAME}"
python3 scripts/standalone/print_benchmark_averages.py "${OUTPUT_FILE_NAME}" >> "${BENCHMARK_AVERAGES_FILE_NAME}"
printf '\n' >> "${BENCHMARK_AVERAGES_FILE_NAME}"
