#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch

results_folder_name="$1"

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS
## adjust this accordingly - by default it is warn
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

input_files=(
  "${input_folder}/FASTA1GB.fna"
  "${input_folder}/FASTQ1GB.fnq"
  "${input_folder}/365.fna"
  "${input_folder}/combined_input.txt"
)

output_files=(
  "${results_folder}/365.txt"
  "${results_folder}/FASTA1GB.txt"
  "${results_folder}/FASTQ1GB.txt"
  "${input_folder}/combined_output.txt"
)

sbwt_file="${input_folder}/coli3.sbwt"
batches=(1 2 3 4 5 6 10 30 70 100 200 1000)

stdoutput="${results_folder}/benchmark_out.txt"

printing_modes=(
  "binary"
  "bool"
  "ascii"
)

# populate combined_input.txt
printf "" > "${input_folder}/combined_input.txt"
printf "" > "${input_folder}/combined_output.txt"
for (( i=0; i<${#input_files[@]}-1; i++ )); do
  printf "${input_files[i]}\n" >> "${input_folder}/combined_input.txt"
  printf "file_${i}\n" >> "${input_folder}/combined_output.txt"
done

echo Running at ${results_folder_name}
for batch in ${batches[@]}; do
  for (( i=0; i<${#input_files[@]}; i++ )); do
    for (( p=0; p<${#printing_modes[@]}; p++ )); do
      echo Now running: File ${input_files[i]} with ${batch} batches in ${printing_modes[p]} format
      echo Now running: File ${input_files[i]} with ${batch} batches in ${printing_modes[p]} format >> ${stdoutput}
      printf "Input file size: " >> ${stdoutput}
      ls -lh ${input_files[i]} | awk '{print  $5}' >> ${stdoutput}
      ./build/bin/sbwt_search index -i ${sbwt_file} -q ${input_files[i]} -o ${output_files[i]} -b ${batch} -c ${printing_modes[p]} >> ${stdoutput}
      printf "Output file size: " >> ${stdoutput}
      ls -lh ${output_files[i]} | awk '{print  $5}' >> ${stdoutput}
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
