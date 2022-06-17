!#/bin/bash
#SBATCH --job-name=BenchmarkQueryReader
#SBATCH --account=project_ay6282
#SBATCH --time=02:00:00

#SBATCH --mem-per-cpu=2G
#SBATCH --partition=small
##SBATCH --mail-type=BEGIN #uncomment to enable mail

module purge
