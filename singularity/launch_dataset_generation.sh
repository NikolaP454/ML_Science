#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=dataset_generation
#SBATCH --output=outputs/dataset_%j.out
#SBATCH --error=errors/dataset_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=gpgpu01

SIF_IMAGE="ml4science.sif"
SCRIPT_PATH="../src/create_data.py"
TORCH_VERSION="2.3.1"   # 2.3.1 Required due to OGB not supporting >=2.6.0

# --- Execute the script within the Singularity container ---
singularity exec --nv $SIF_IMAGE uv run \
    --with torch==$TORCH_VERSION $SCRIPT_PATH  "$@"