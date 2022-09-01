# Run the benchmark_main several times and save the results to a file


mkdir -p results
input_files=(
  "benchmark_objects/FASTA1GB.fna"
  "benchmark_objects/FASTQ1GB.fnq"
  "test_objects/365.fna"
  "combined.txt"
)
output_files=(
  "results/FASTA1GB.txt"
  "results/FASTQ1GB.txt"
  "results/365.txt"
  "results/combined_output.txt"
)
sbwt_file="test_objects/parsed100.sbwt"
batches=(1, 3, 10, 30, 70, 100, 200, 1000)

for batch in ${batches[@]}; do
for (( i=0; i<${#input_files[@]}; i++ )); do
echo Now running file ${input_file[i]} with ${batch} batches
./build/bin/main_cuda -i ${sbwt_file} -q ${input_files[i]} -o ${output_files[i]} -b ${batch} -u 429496729600
done
done
