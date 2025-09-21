#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=llm_finetuning
#SBATCH --output=outputs/train_%j.out
#SBATCH --error=errors/train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=gpgpu01

SIF_IMAGE="ml4science.sif"
SCRIPT_PATH="../src/train_model.py"
UV_GROUP="finetuning"

# --- Execute the script within the Singularity container ---
singularity exec --nv $SIF_IMAGE uv run \
    --group $UV_GROUP $SCRIPT_PATH  "$@"