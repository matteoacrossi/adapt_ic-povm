sbatch --array=1,3,16,23 --mem-per-cpu=2GB --job-name=pauli_jw run_simulation.sh \
                                    --method Pauli \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=1,3,16,23 --mem-per-cpu=2GB --job-name=grpauli_jw run_simulation.sh \
                                    --method Grouped_Pauli \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=1,3,16,23 --mem-per-cpu=4GB --job-name=sic_jw run_simulation.sh \
                                    --method SIC-POVM \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=1,3,16,23 --mem-per-cpu=4GB --job-name=goog_jw run_simulation.sh \
                                    --method Google-POVM \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=1,3,16,23 --mem-per-cpu=4GB --job-name=grad_sic_jw run_simulation.sh \
                                    --method Grad-POVM \
                                    --samples 1 \
                                    --shots 1000000

sbatch --array=1,3,16,23 --mem-per-cpu=4GB --job-name=grad_goog_jw run_simulation.sh \
                                    --method Grad-Google-POVM \
                                    --samples 1 \
                                    --shots 1000000

# 14-qubit job

sbatch --array=28 --mem-per-cpu=10GB --job-name=pauli_jw run_simulation.sh \
                                    --method Pauli \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=28 --mem-per-cpu=10GB --job-name=grpauli_jw run_simulation.sh \
                                    --method Grouped_Pauli \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=28 --mem-per-cpu=10GB --job-name=sic_jw run_simulation.sh \
                                    --method SIC-POVM \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=28 --mem-per-cpu=10GB --job-name=goog_jw run_simulation.sh \
                                    --method Google-POVM \
                                    --samples 1 \
                                    --shots 1000 10000 10000 1000000

sbatch --array=28 --mem-per-cpu=10GB --job-name=grad_sic_jw run_simulation.sh \
                                    --method Grad-POVM \
                                    --samples 1 \
                                    --shots 1000000

sbatch --array=28 --mem-per-cpu=10GB --job-name=grad_goog_jw run_simulation.sh \
                                    --method Grad-Google-POVM \
                                    --samples 1 \
                                    --shots 1000000