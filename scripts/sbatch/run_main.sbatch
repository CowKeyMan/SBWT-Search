#!/bin/bash
## Meant to be run on Mahti supercomputer by CSC: https://research.csc.fi/-/mahti

#SBATCH --job-name=Main
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
#SBATCH --gres=gpu:a100:1,nvme:100
## Remove unnecessary modules
module purge
## Load in-build modules
module load cmake
module load gcc
module load cuda
module load python-data
module load bzip2

export OLD_PWD="${PWD}"
mkdir -p "main_results"

cp -r . "${LOCAL_SCRATCH}/SBWT-Search/"
cd "${LOCAL_SCRATCH}/SBWT-Search"

rm -rf build

sh scripts/build/release.sh
/usr/bin/time ./build/bin/main_cuda -i test_objects/parsed100.sbwt -q test_objects/queries.fna -o main_results/output.txt -u 429496729600
/usr/bin/time ./build/bin/main_cuda -i test_objects/parsed100.sbwt -q test_objects/fasta.fasta -o main_results/output.txt -u 429496729600
/usr/bin/time ./build/bin/main_cuda -i test_objects/parsed100.sbwt -q test_objects/fastq.fastq -o main_results/output.txt -u 429496729600

cd "${OLD_PWD}"
cp -r main_results "${LOCAL_SCRATCH}/SBWT-Search/main_results"
