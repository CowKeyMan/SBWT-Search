#!/bin/bash
## Meant to be run on Mahti supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Main
## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=dongelr1
## Maximum time
#SBATCH --time=01:00:00
## put stderr to benchmark_err.txt
#SBATCH --error err.txt
## put stdout to out.txt
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
#SBATCH --partition=gputest
## We only want a single a100 gpu and 10GB of NVME storage
#SBATCH --gres=gpu:a100:1,nvme:300

## Remove unnecessary modules
module purge
## Load in modules
module load gcc cuda python-data bzip2

## We want to perform our work in LOCAL SCRATCh
export OLD_PWD="${PWD}"
cp -r . "${LOCAL_SCRATCH}/SBWT-Search/"
cd "${LOCAL_SCRATCH}/SBWT-Search"

rm -rf build

## Build
sh scripts/build/release.sh

sh scripts/standalone/run_benchmarks.sh

cd "${OLD_PWD}"
cp -r "${LOCAL_SCRATCH}/SBWT-Search/results" .
