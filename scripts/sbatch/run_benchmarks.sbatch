#!/bin/bash

## Meant to be run on Mahti supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Benchmark

## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=dongelr1

## Maximum time
#SBATCH --time=00:15:00

## put stderr to err.txt
#SBATCH --error err.txt

## put stdout to out.txt
#SBATCH --output out.txt

## We only need a single node
#SBATCH --nodes 1

## Number of cores per node. We request 8 for faster compilation
#SBATCH --ntasks-per-node=8

## Since this is a benchmarking script, we do not want it to share resources
##SBATCH --exclusive

## Amount of memory per node
#SBATCH --mem-per-cpu=2G

## This partition on mahti has NVME
#SBATCH --partition=gpusmall

## We only want a single a100 gpu and 10GB of NVME storage
#SBATCH --gres=gpu:a100:0,nvme:10

## Remove unnecessary modules
module purge

## Load in-build modules
module load cmake
module load gcc
module load cuda
module load python-data
module load bzip2

export OLD_PWD="${PWD}"

cp -r . "${LOCAL_SCRATCH}/SBWT-Search/"
cd "${LOCAL_SCRATCH}/SBWT-Search"

rm "build/CMakeCache.txt"

rm -rf build

sh scripts/build/build_benchmarks.sh
sh scripts/standalone/run_benchmarks.sh

cd "${OLD_PWD}"

mkdir -p "benchmark_results"
cat "${LOCAL_SCRATCH}/SBWT-Search/benchmark_results/averages.txt" >> "benchmark_results/averages.txt"
rm  "${LOCAL_SCRATCH}/SBWT-Search/benchmark_results/averages.txt"
cp  "${LOCAL_SCRATCH}/SBWT-Search/benchmark_results/"* "benchmark_results"
