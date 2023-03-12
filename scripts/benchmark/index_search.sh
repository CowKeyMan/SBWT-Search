#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch

if [ $# -ne 1 ]; then
  echo "Please enter a single argument to this script, which is the output folder"
  exit 1
fi

results_folder_name="$1"

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS
## Set loglevel to trace because we use this to get timing statistics
export SPDLOG_LEVEL=TRACE

# top_level_directory="/dev/shm"
top_level_directory="."
input_folder="${top_level_directory}/benchmark_objects"
results_folder="${top_level_directory}/benchmark_results/${results_folder_name}"

mkdir -p "${results_folder}"

if [ "${top_level_directory}" != "." ]; then
  cp -r "benchmark_objects" "${input_folder}"
  cp -r "benchmark_results/${results_folder_name}/" "${top_level_directory}/benchmark_results/${results_folder_name}/"
fi

if [ ! -d "benchmark_objects/index" ] || [ ! -f "benchmark_objects/index/index.tdbg" ] || [ ! -f "benchmark_objects/index/index.tcolors" ]; then
  echo "This script expects a 'benchmark_objects/index' folder with 'index.tdbg' and 'index.tcolors' to be present within that folder"
  exit 1
fi

dbg_file="${input_folder}/index/index.tdbg"

out_file="${results_folder}/benchmark_out.txt"

printing_modes=(
  "binary"
  "ascii"
)

input_files=(
  "${input_folder}/combined_reads_zipped.list"
  "${input_folder}/combined_reads_unzipped.list"
)
output_file="${input_folder}/combined_indexes_output.list"
printf "" > ${out_file}

devices=(
  "nvidia"
  "cpu"
)

streams_options=({1..8})

echo Running at ${results_folder_name}
for device in ${devices[@]}; do
  ./scripts/build/release_${device}.sh >&2
  for streams in ${streams_options[@]}; do
    for (( i=0; i<${#input_files[@]}; i++ )); do
      for (( p=0; p<${#printing_modes[@]}; p++ )); do
        echo Now running: File ${input_files[i]} with ${streams} streams in ${printing_modes[p]} format on ${device} device
        echo Now running: File ${input_files[i]} with ${streams} streams in ${printing_modes[p]} format on ${device} device >> ${out_file}
        ./build/bin/sbwt_search index -i ${dbg_file} -q ${input_files[i]} -o ${output_file} -s ${streams} -c ${printing_modes[p]} >> ${out_file}
      done
    done
  done
done

if [ "${top_level_directory}" != "." ]; then
  cp -r "${results_folder}" "benchmark_results/${results_folder_name}"
fi

if [ "${top_level_directory}" == "/dev/shm" ]; then
  rm -r "/dev/shm/benchmark_objects"
  rm -r "/dev/shm/benchmark_results"
fi
