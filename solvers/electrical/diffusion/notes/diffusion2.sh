#!/bin/bash
# --------------- SLURM Parameters ---------------
#SBATCH --qos=normal
#SBATCH --ntasks=39
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=48:00:00
#SBATCH --job-name=sympy2a
#SBATCH --mail-user=telegram:610987613
#SBATCH --mail-type=ALL
#SBATCH --output=sympy2-%j.out
#SBATCH --error=sympy2-%j.out

# --------------- Load Modules -------------------
module load mpich

# --------------- Commands -----------------------
srun python3 -u diffusion2.py
