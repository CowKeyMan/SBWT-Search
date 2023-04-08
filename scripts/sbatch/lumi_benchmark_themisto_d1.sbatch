#!/bin/bash

# This will configure items for benchmarking and then run the benchmarks
# through the scripts/standalone/benchmark.sh script. Meant to be run on Mahti
# supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=SBWT_themisto_d1
#SBATCH --account=project_462000116
#SBATCH --error themisto_err.txt
#SBATCH --output themisto_out.txt
#SBATCH --nodes 1
#SBATCH --gres=gpu:1

## Maximum time
#SBATCH --time=03:00:00
#SBATCH --partition=standard-g

## Load in modules
## Note: on lumi, you will need to download and install your own cmake
module swap craype-x86-rome craype-x86-trento
module load gcc rocm craype-accel-amd-gfx90a cray-python

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS

chmod +777 scripts/**/*
export DATETIME="$(date +"%Y-%m-%d_%H-%M-%S_%z")"
export OUTPUT_FOLDER="themisto{DATETIME}"
## We want to perform our work in LOCAL_SCRATCH
### We want to perform our work in LOCAL_SCRATCH
mkdir -p "/flash/project_462000116/cauchida/themisto_d20"
export LOCAL_SCRATCH="/flash/project_462000116/cauchida/themisto_d20"
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
bash ./scripts/build/release.sh nvidia >&2
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
  --n-threads 128 \
  --sort-output \
  --temp-dir themisto_temp \
  --out-file-list benchmark_objects/list_files/output/color_search_results_running.list \
  --query-file-list benchmark_objects/list_files/input/unzipped_seqs.list \
  --index-prefix benchmark_objects/index/index \
  --verbose \
  --threshold 0.7

cd "${OLD_PWD}"

rm -rf ${LOCAL_SCRATCH}
