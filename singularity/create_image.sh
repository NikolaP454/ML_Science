#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=build_singularity
#SBATCH --output=outputs/build_singularity.out
#SBATCH --error=errors/build_singularity.err
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=gpgpu01

# Load Singularity
module load singularity

# Build the image
singularity build --fakeroot ml4science.sif ml4science.def
