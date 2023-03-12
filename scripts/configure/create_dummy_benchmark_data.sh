#!/bin/bash

# This script is used to test benchmarking locally. We take the files from the pipelines tests and benchmarks with them, to see that the benchmarking scripts are working appropriately.

mkdir -p benchmark_objects
cd benchmark_objects

# copy files and make zipped copies
cp ../test_objects/full_pipeline/index_search/*.fna .
cp ../test_objects/full_pipeline/index_search/*.fnq .

uncompressed_files=`ls *.fna *.fnq`
for file in ${uncompressed_files}; do
  yes n | gzip -k ${file}
done

cd ..

function zipped_files() {
  find benchmark_objects/*.fnq.gz benchmark_objects/*.fna.gz
}

zipped_files > benchmark_objects/combined_reads_zipped.list
zipped_files | sed -e 's/\.fnq\.gz$/.fnq/' | sed -e 's/\.fna\.gz$/.fna/' > benchmark_objects/combined_reads_unzipped.list
zipped_files | sed -e 's/\.fnq\.gz$/.indexes/' | sed -e 's/\.fna\.gz$/.indexes/' > benchmark_objects/combined_indexes_output.list

zipped_files | sed -e 's/\.fnq\.gz$/.indexes.txt/' | sed -e 's/\.fna\.gz$/.indexes.txt/' > benchmark_objects/combined_indexes_ascii.list
zipped_files | sed -e 's/\.fnq\.gz$/.indexes.bin/' | sed -e 's/\.fna\.gz$/.indexes.bin/' > benchmark_objects/combined_indexes_binary.list

zipped_files | sed -e 's/\.fnq\.gz$/.colors.txt/' | sed -e 's/\.fna\.gz$/.colors.txt/' > benchmark_objects/combined_colors_ascii.list
zipped_files | sed -e 's/\.fnq\.gz$/.colors.csv/' | sed -e 's/\.fna\.gz$/.colors.csv/' > benchmark_objects/combined_colors_csv.list
zipped_files | sed -e 's/\.fnq\.gz$/.colors.bin/' | sed -e 's/\.fna\.gz$/.colors.bin/' > benchmark_objects/combined_colors_binary.list

cd ..
