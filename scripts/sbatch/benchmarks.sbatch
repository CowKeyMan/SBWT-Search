#!/bin/bash

# This will configure items for benchmarking and then run the benchmarks
# through the scripts/standalone/benchmark.sh script. Meant to be run on Mahti
# supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Main
## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=dongelr1
## put stderr to benchmark_err.txt
#SBATCH --error benchmark_err.txt
## put stdout to benchmark_out.txt
#SBATCH --output benchmark_out.txt
## We only need a single node
#SBATCH --nodes 1
## Number of tasks per node.
#SBATCH --ntasks-per-node=1
## Number of cpus per task.
#SBATCH --cpus-per-task=64
## Since this is a benchmarking script, we might not want it to share resources
##SBATCH --exclusive
## Amount of memory per node
#SBATCH --mem-per-cpu=29G
## We only want a single a100 gpu and 10GB of NVME storage
#SBATCH --gres=gpu:a100:1,nvme:300

## The partitions on mahti with nvme and maximum time
##SBATCH --partition=gputest
##SBATCH --time=00:15:00
## Maximum time
#SBATCH --time=03:00:00
#SBATCH --partition=gpusmall

## Load in modules
module load cmake gcc cuda python-data bzip2 git

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
bash ./scripts/build/release.sh &
cp -r "${OLD_PWD}/benchmark_objects/" "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects" &
wait < <(jobs -p) # wait for jobs to finish
t2=$(date +%s%3N)

mkdir -p "benchmark_results/${OUTPUT_FOLDER}"
echo "Time taken to copy and build in LOCAL_SCRATCH: $((t2-t1)) ms" >> "benchmark_results/${OUTPUT_FOLDER}/benchmark_out.txt"

sh -x scripts/standalone/benchmark.sh "${OUTPUT_FOLDER}"

cd "${OLD_PWD}"
cp -r "${LOCAL_SCRATCH}/SBWT-Search/benchmark_results" .
