#!/bin/bash
#SBATCH --mail-user=matteo.rossi@utu.fi
#SBATCH --mail-type=END,FAIL
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --partition=all
#SBATCH --job-name=tomo_chem
#SBATCH --time=96:00:00
#SBATCH --output=logs/%A_%a.out

source env/bin/activate

python tomography_script.py