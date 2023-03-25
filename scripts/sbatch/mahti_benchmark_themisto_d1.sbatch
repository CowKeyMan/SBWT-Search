#!/bin/bash

# This will configure items for benchmarking and then run the benchmarks
# through the scripts/standalone/benchmark.sh script. Meant to be run on Mahti
# supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=themisto_pseudoalign
#SBATCH --account=dongelr1
#SBATCH --error themisto_d1_err.txt
#SBATCH --output themisto_d1_out.txt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
##SBATCH --exclusive
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=nvme:350

##SBATCH --partition=gputest
##SBATCH --time=00:15:00

#SBATCH --time=10:00:00
#SBATCH --partition=gpusmall

## Load in modules
module load cmake gcc cuda python-data bzip2 git

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS

chmod +777 scripts/**/*
export DATETIME="$(date +"%Y-%m-%d_%H-%M-%S_%z")"
export OUTPUT_FOLDER="${DATETIME}"
## We want to perform our work in LOCAL_SCRATCH
export OLD_PWD="${PWD}"
mkdir -p "${LOCAL_SCRATCH}/SBWT-Search/"
cd "${LOCAL_SCRATCH}/SBWT-Search"
rm -rf build

# time the copy and build
t1=$(date +%s%3N)
cp -r "${OLD_PWD}/src/" "${LOCAL_SCRATCH}/SBWT-Search/src" &
cp -r "${OLD_PWD}/cmake/" "${LOCAL_SCRATCH}/SBWT-Search/cmake" &
cp -r "${OLD_PWD}/scripts/" "${LOCAL_SCRATCH}/SBWT-Search/scripts" &
cp "${OLD_PWD}/CMakeLists.txt" "${LOCAL_SCRATCH}/SBWT-Search/CMakeLists.txt" &
wait < <(jobs -p) # wait for jobs to finish
cp -r "${OLD_PWD}/benchmark_objects/" "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects" &
bash ./scripts/build/release_nvidia.sh >&2
wait < <(jobs -p) # wait for jobs to finish
t2=$(date +%s%3N)

echo "Time taken to copy and build in LOCAL_SCRATCH: $((t2-t1)) ms"

# get themisto executable
cd benchmark_objects
wget -nc https://github.com/algbio/themisto/releases/download/3.0.0/themisto_linux-v3.0.0.tar.gz
tar -xvzf themisto_linux-v3.0.0.tar.gz
cd ..
mkdir -p themisto_temp

cp benchmark_objects/index/index_d1.tcolors benchmark_objects/index/index.tcolors

./benchmark_objects/themisto_linux-v3.0.0/themisto pseudoalign \
  --n-threads 32 \
  --sort-output \
  --temp-dir themisto_temp \
  --out-file-list benchmark_objects/list_files/output/color_search_results.list \
  --query-file-list benchmark_objects/list_files/input/unzipped_seqs.list \
  --index-prefix benchmark_objects/index/index \
  --verbose \
  --threshold 0.7

cd "${OLD_PWD}"
