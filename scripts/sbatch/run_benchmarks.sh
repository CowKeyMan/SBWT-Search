!/bin/bash
#SBATCH --job-name=BenchmarkQueryReader

## The account to charge. Look at the unix group from https://my.csc.fi/myProjects
#SBATCH --account=${PROJECT}

## Maximum time
#SBATCH --time=02:00:00

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

## We only want a single a100 gpu and 4GB of NVME storage
#SBATCH --gres=gpu:a100:1,nvme:4

## Remove unnecessary modules
module purge

## Load in-build modules
module load cmake
module load gcc
module load cuda
module load python-data
module load bzip2

sh scripts/build/build_benchmarks.sh
sh scripts/standalone/run_benchmarks.sh
