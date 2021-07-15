# BK mapping LiH
sbatch --array=10 --partition=all --job-name=grad_povm run_simulation_counts.sh Grad-POVM
sbatch --array=10 --partition=all -c1 --job-name=sic_povm run_simulation_counts.sh SIC-POVM
sbatch --array=10 --partition=all --job-name=grad_povm run_simulation_counts.sh Grad-Google-POVM
sbatch --array=10 --partition=all -c1 --job-name=sic_povm run_simulation_counts.sh Google-POVM

# Neven mapping H2O
sbatch --array=19 --partition=all --job-name=grad_povm run_simulation_counts.sh Grad-POVM
sbatch --array=19 --partition=all -c1 --job-name=sic_povm run_simulation_counts.sh SIC-POVM
sbatch --array=19 --partition=all --job-name=grad_povm run_simulation_counts.sh Grad-Google-POVM
sbatch --array=19 --partition=all -c1 --job-name=sic_povm run_simulation_counts.sh Google-POVM