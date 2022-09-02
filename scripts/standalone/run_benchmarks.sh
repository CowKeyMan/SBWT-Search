# Run the benchmark_main several times and save the results to a file

DATETIME="$(date +"%Y-%m-%d_%I-%M-%S_%p")"

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS
## adjust this accordingly - by default it is warn
export SPDLOG_LEVEL=TRACE

results_folder="results/${DATETIME}"

mkdir -p results
mkdir -p ${results_folder}


input_files=(
  "benchmark_objects/FASTA1GB.fna"
  "benchmark_objects/FASTQ1GB.fnq"
  "test_objects/365.fna"
  "benchmark_objects/combined_input.txt"
)

# populate combined_input.txt
printf "" > "benchmark_objects/combined_input.txt"
printf "" > "benchmark_objects/combined_output.txt"
for (( i=0; i<${#input_files[@]}-1; i++ )); do
printf "${input_files[i]}\n" >> "benchmark_objects/combined_input.txt"
printf "file_${i}\n" >> "benchmark_objects/combined_output.txt"
done

output_files=(
  "${results_folder}/FASTA1GB.txt"
  "${results_folder}/FASTQ1GB.txt"
  "${results_folder}/365.txt"
  "benchmark_objects/combined_output.txt"
)
sbwt_file="data/coli3.sbwt"
batches=(1 3 10 30 70 100 200 1000)

stdoutput="${results_folder}/benchmark_out.txt"

for batch in ${batches[@]}; do
for (( i=0; i<${#input_files[@]}; i++ )); do
echo Now running: File ${input_files[i]} with ${batch} batches >> ${stdoutput}
./build/bin/main_cuda -i ${sbwt_file} -q ${input_files[i]} -o ${output_files[i]} -b ${batch} >> ${stdoutput}
done
done
