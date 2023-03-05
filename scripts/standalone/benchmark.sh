#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch

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

dbg_file="${input_folder}/ecoli_index/index.tdbg"
colors_file="${input_folder}/ecoli_index/index.tcolors"

out_file="${results_folder}/benchmark_out.txt"

batches=(1)

index_printing_modes=(
  "binary"
  "bool"
  "ascii"
)

colors_printing_modes=(
  "binary"
  "bool"
  "ascii"
)

index_input_files=(
  "${input_folder}/combined_reads_zipped.txt"
  "${input_folder}/combined_reads_unzipped.txt"
)
index_output_file="${input_folder}/combined_indexes_output.txt"

echo Running at ${results_folder_name}
for batch in ${batches[@]}; do
  for (( i=0; i<${#index_input_files[@]}; i++ )); do
    for (( p=0; p<${#index_printing_modes[@]}; p++ )); do
      echo Now running: File ${index_input_files[i]} with ${batch} batches in ${index_printing_modes[p]} format
      echo Now running: File ${index_input_files[i]} with ${batch} batches in ${index_printing_modes[p]} format >> ${out_file}
      ./build/bin/sbwt_search index -i ${dbg_file} -q ${index_input_files[i]} -o ${index_output_file} -b ${batch} -c ${index_printing_modes[p]} >> ${out_file}
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
