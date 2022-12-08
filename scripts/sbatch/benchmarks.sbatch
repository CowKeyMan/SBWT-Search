#!/bin/bash
## Meant to be run on Mahti supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Main
## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=dongelr1
## Maximum time
#SBATCH --time=03:00:00
## put stderr to benchmark_err.txt
#SBATCH --error benchmark_err.txt
## put stdout to benchmark_out.txt
#SBATCH --output benchmark_out.txt
## We only need a single node
#SBATCH --nodes 1
## Number of tasks per node.
#SBATCH --ntasks-per-node=1
## Number of cpus per task.
#SBATCH --cpus-per-task=16
## Since this is a benchmarking script, we might not want it to share resources
##SBATCH --exclusive
## Amount of memory per node
#SBATCH --mem-per-cpu=29G
## This partition on mahti has NVME
#SBATCH --partition=gpusmall
## We only want a single a100 gpu and 10GB of NVME storage
#SBATCH --gres=gpu:a100:1,nvme:300

## Remove unnecessary modules
module purge
## Load in modules
module load cmake gcc cuda python-data bzip2 git

chmod +777 scripts/**/*
DATETIME="$(date +"%Y-%m-%d_%H-%M-%S_%z")"
## We want to perform our work in LOCAL_SCRATCH
export OLD_PWD="${PWD}"
# time the copy
t1=$(date +%s%3N)
mkdir -p "${LOCAL_SCRATCH}/SBWT-Search/build/bin"
cp "${PWD}/CMakeLists.txt" "${LOCAL_SCRATCH}/SBWT-Search/CMakeLists.txt"
cp -r "${PWD}/src/" "${LOCAL_SCRATCH}/SBWT-Search/src"
mkdir "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects"
cp "${PWD}/benchmark_objects/"*.fnq "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects"
cp "${PWD}/benchmark_objects/"*.fna "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects"
cp "${PWD}/benchmark_objects/"*.sbwt "${LOCAL_SCRATCH}/SBWT-Search/benchmark_objects"
cp -r "${PWD}/cmake/" "${LOCAL_SCRATCH}/SBWT-Search/cmake"
cp -r "${PWD}/scripts/" "${LOCAL_SCRATCH}/SBWT-Search/scripts"
t2=$(date +%s%3N)

cd "${LOCAL_SCRATCH}/SBWT-Search"
mkdir -p "benchmark_results/${DATETIME}"
echo "Time taken to copy to LOCAL_SCRATCH: $((t2-t1)) ms" >> "benchmark_results/${DATETIME}/benchmark_out.txt"

rm -rf build

## Build
sh scripts/build/release.sh a

sh -x scripts/standalone/benchmark.sh "${DATETIME}"

cd "${OLD_PWD}"
cp -r "${LOCAL_SCRATCH}/SBWT-Search/benchmark_results" .
