#!/bin/bash
#SBATCH --mail-user=matteo.rossi@utu.fi
#SBATCH --mail-type=END,FAIL
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=96:00:00
#SBATCH --output=logs/%j.out

source ./env/bin/activate
python prepare_h_chain_hamiltonians.py
