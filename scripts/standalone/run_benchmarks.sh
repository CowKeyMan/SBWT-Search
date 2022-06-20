# Run the benchmark_main several times and save the results to a file


mkdir -p benchmark_results

DATETIME="$(date +"%Y_%m_%d_%I:%M:%S_%p")"
OUTPUT_FILE_NAME="benchmark_results/${DATETIME}_benchmark"

# Run once so that we do not get any weird items taking longer than they should
build/src/benchmark_main > /dev/null 2>/dev/null
for i in {1..20}
do
  build/src/benchmark_main >> "${OUTPUT_FILE_NAME}.txt" 2>/dev/null
  printf '\n' >>"${OUTPUT_FILE_NAME}.txt"
done

python3 scripts/standalone/print_benchmark_averages.py "${OUTPUT_FILE_NAME}.txt" > "${OUTPUT_FILE_NAME}_averages.txt"
