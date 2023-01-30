#!/bin/bash

# Runs the executable on the specified parameters. Meant to be run on Mahti
# supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Main
## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=dongelr1
## Maximum time
#SBATCH --time=00:15:00
## put stderr to main_err.txt
#SBATCH --error main_err.txt
## put stdout to main_out.txt
#SBATCH --output main_out.txt
## We only need a single node
#SBATCH --nodes 1
## Number of tasks per node.
#SBATCH --ntasks-per-node=1
## Number of cpus per task.
#SBATCH --cpus-per-task=16
## Since this is a benchmarking script, we do not want it to share resources
##SBATCH --exclusive
## Amount of memory per node
#SBATCH --mem-per-cpu=29G
## This partition on mahti has NVME
#SBATCH --partition=gputest
## We only want a single a100 gpu and 10GB of NVME storage
#SBATCH --gres=gpu:a100:1,nvme:300

## Remove unnecessary modules
module purge
## Load in modules
module load cmake gcc cuda python-data bzip2

## By default this is 1, which would halt our program
unset OMP_NUM_THREADS
## adjust this accordingly - by default it is warn
export SPDLOG_LEVEL=WARN

## We want to perform our work in LOCAL SCRATCh
export OLD_PWD="${PWD}"
cp -r . "${LOCAL_SCRATCH}/SBWT-Search/"
cd "${LOCAL_SCRATCH}/SBWT-Search"

# Initialise results folder
DATETIME="$(date +"%Y-%m-%d_%H-%M-%S_%z")"
mkdir -p main_results
results_folder="main_results/${DATETIME}"
mkdir -p ${results_folder}

rm -rf build

## Build
sh scripts/build/release.sh

input_files=(
  "benchmark_objects/365.fna"
)
output_files=(
  "${results_folder}/365.txt"
)
sbwt_file="benchmark_objects/coli3.sbwt"
batches=30

for (( i=0; i<${#input_files[@]}; i++ )); do
echo Now running: File ${input_files[i]} with ${batch} batches
/usr/bin/time ./build/bin/sbwt_search index -i ${sbwt_file} -q ${input_files[i]} -o ${output_files[i]} -b ${batches}
done

cd "${OLD_PWD}"
cp -r "${LOCAL_SCRATCH}/SBWT-Search/main_results" .
