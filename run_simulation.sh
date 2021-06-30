#!/bin/bash
#SBATCH --mail-user=matteo.rossi@utu.fi
#SBATCH --mail-type=END,FAIL
#SBATCH -c 1
#SBATCH -n 100
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=96:00:00
#SBATCH --output=logs/%A_%a.out

source env/bin/activate

srun -l --ntasks $SLURM_NTASKS --unbuffered python run_simulation.py \
                                   --hamiltonian $SLURM_ARRAY_TASK_ID \
                                   --outfile raw_data/raw_data_$SLURM_ARRAY_JOB_ID.txt \
                                   --file hamiltonians.pickle \
                                   $@
