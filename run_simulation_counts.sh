#!/bin/bash
#SBATCH --mail-user=matteo.rossi@utu.fi
#SBATCH --mail-type=END,FAIL
#SBATCH -c 4
#SBATCH -n 20
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=96:00:00
#SBATCH --output=logs/%A_%a.out

source env/bin/activate

srun -l --ntasks $SLURM_NTASKS --unbuffered python run_simulation.py \
                                   --hamiltonian $SLURM_ARRAY_TASK_ID \
                                   --file hamiltonians.pickle \
                                   --outfile data/counts_data_3.txt \
                                   --samples 1 \
                                   --shots 100000 \
                                   --method $1 \
                                   --counts
                                   $2
