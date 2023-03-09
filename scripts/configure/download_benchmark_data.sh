#!/bin/bash

# Download the data used for benchmarking from public databases. The dataset
# file list is contained `scripts/configure/benchmark_datra_list.txt`.  This
# file contains all the files obtained from
# https://www.ebi.ac.uk/ena/browser/view/PRJEB32631, from this paper:
# doi:10.1038/s41467-022-35178-5.  The original dataset is 2TB big, so I have
# chosen to only have the first 20 files. The index contains ecoli genomes,
# found at https://zenodo.org/record/6656897#.ZASdeR9BwQ8, created using
# themisto using the genomes compiled in these papers:
# https://doi.org/10.1099/mgen.0.000499, https://doi.org/10.1099/mgen.0.000499
# and https://doi.org/10.1038/s41586-019-1560-1


mkdir -p benchmark_objects
cd benchmark_objects

wget -nc -i ../scripts/configure/benchmark_data_list.txt
# "yes n" skips files if they exist as unzipped already
yes n | gunzip -k *fastq.gz

# if directory exists then it has already been downloaded
if [ ! -d "ecoli_index" ]; then
  mkdir -p ecoli_index
  cd ecoli_index
  head -20 scripts/configure/full_benchmark_data_list.txt | wget -nc -i -
  tar -xvf E_coli_lineage_index_v1-0-0.tar.gz
  mv wrk/users/temaklin/cocov2-reference-sequences/E_col/E_col_index/index.tdbg .
  mv wrk/users/temaklin/cocov2-reference-sequences/E_col/E_col_index/index.tcolors .
  mv wrk/users/temaklin/cocov2-reference-sequences/E_col/E_col_mSWEEP_indicators.txt .
  rm -r wrk
fi

cd ../..

find benchmark_objects/*.fastq.gz > benchmark_objects/combined_reads_zipped.txt
find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.fastq/' > benchmark_objects/combined_reads_unzipped.txt
find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.indexes/' > benchmark_objects/combined_indexes_output.txt

find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.indexes.txt/' > benchmark_objects/combined_indexes_ascii.txt
find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.indexes.bin/' > benchmark_objects/combined_indexes_binary.txt

find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.colors.txt/' > benchmark_objects/combined_colors_ascii.txt
find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.colors.csv/' > benchmark_objects/combined_colors_csv.txt
find benchmark_objects/*.fastq.gz | sed -e 's/\.fastq\.gz$/.colors.bin/' > benchmark_objects/combined_colors_binary.txt

cd ..
