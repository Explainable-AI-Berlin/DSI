#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/job.out.%A_%a
#SBATCH -e ./log/job.err.%A_%a
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J intervention_effects
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000

#SBATCH --time=1:00:00

module purge
module load apptainer

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_TOKEN=$(cat ./.hf_token)
export PYTHONPATH=$(pwd):$PYTHONPATH

srun \
    apptainer exec DSI.sif \
    python main_results/experiments.py \
    --steps intervention_effects
