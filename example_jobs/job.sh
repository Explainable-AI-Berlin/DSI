#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log/job.out.%A_%a
#SBATCH -e ./log/job.err.%A_%a
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J iterative_array
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000

#SBATCH --time=8:00:00

#SBATCH --array=0-35

module purge
module load apptainer

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_TOKEN=$(cat ./.hf_token)
export PYTHONPATH=$(pwd):$PYTHONPATH

MODELS=(Qwen/Qwen3-8B google/gemma-3-4b-it meta-llama/Llama-3.2-3B-Instruct)
TASKS=(capitalize country-capital english-french present-past sentiment product-company \
       fruit_v_animal_3 color_v_animal_3 object_v_concept_3 choose_first_of_3 \
       adjective_v_verb_3 choose_middle_of_3)

N_TASKS=${#TASKS[@]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / N_TASKS))]}
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID % N_TASKS))]}

srun \
    apptainer exec DSI.sif \
    python main_results/experiments.py \
    --model-names $MODEL \
    --tasks $TASK \
    --add-resid-dim \
    --steps baseline attributions compute_iterative eval_iterative no_robustification no_refinement