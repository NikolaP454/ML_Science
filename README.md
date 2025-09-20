# Scientific Expansion using ML

This project aims to develop a method using machine learning to expand scientific boundaries, one step at a time.

# Setup

## Singularity Image Creation

To create the singularity image required for running the scripts on the HPC machine you will need to run the following commands:

```sh
cd singularity
singularity build --fakeroot ml4science.sif ml4science.def
```