#!/bin/bash

# Download the data used for benchmarking from public databases. The dataset
# file list is contained `scripts/configure/benchmark_datra_list.txt`. This
# file contains all the files obtained from
# https://www.ebi.ac.uk/ena/browser/view/PRJEB32631, from this paper:
# doi:10.1038/s41467-022-35178-5. The original dataset is 2TB big, so I have
# chosen to only have the first 20 files. The index contains ecoli genomes,
# found at https://zenodo.org/record/6656897#.ZASdeR9BwQ8, created using
# themisto using the genomes compiled in these papers:
# https://doi.org/10.1099/mgen.0.000499, https://doi.org/10.1099/mgen.0.000499
# and https://doi.org/10.1038/s41586-019-1560-1


mkdir -p benchmark_objects
cd benchmark_objects

head -20 ../scripts/configure/full_benchmark_data_list.txt | wget -nc -i -
# "yes n" skips files if they exist as unzipped already
yes n | gunzip -k *fastq.gz

cd ..

function zipped_files() {
  find benchmark_objects/*.fastq.gz
}

zipped_files > benchmark_objects/combined_reads_zipped.list
zipped_files | sed -e 's/\.fastq\.gz$/.fastq/' > benchmark_objects/combined_reads_unzipped.list
zipped_files | sed -e 's/\.fastq\.gz$/.indexes/' > benchmark_objects/combined_indexes_output.list

zipped_files | sed -e 's/\.fastq\.gz$/.indexes.txt/' > benchmark_objects/combined_indexes_ascii.list
zipped_files | sed -e 's/\.fastq\.gz$/.indexes.bin/' > benchmark_objects/combined_indexes_binary.list

zipped_files | sed -e 's/\.fastq\.gz$/.colors.txt/' > benchmark_objects/combined_colors_ascii.list
zipped_files | sed -e 's/\.fastq\.gz$/.colors.csv/' > benchmark_objects/combined_colors_csv.list
zipped_files | sed -e 's/\.fastq\.gz$/.colors.bin/' > benchmark_objects/combined_colors_binary.list

cd ..
