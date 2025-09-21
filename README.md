# Scientific Expansion using ML

This project aims to develop a method using machine learning to expand scientific boundaries, one step at a time.

# Setup

## Singularity Image Creation

To create the singularity image required for running the scripts on the HPC machine you will need to run the following commands:

```sh
cd singularity
singularity build --fakeroot ml4science.sif ml4science.def
```

## Data Source

To run the required dataset generation process you will need to download the titles and abstracts of the papers from the OGB website ([link](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)). After you have downloaded and move the `titleabs.tsv.gz`, you will need to extract the `.tsv` file with the following command:

```sh
gzip -dk titleabs.tsv.gz
```

# Running experiments

## Dataset Generation

To create the dataset required (using singularity) you will need to run the following commands:

```sh
cd singularity

sbatch launch_dataset_generation.sh \
    --experiment_path PATH  \
    --data_path PATH        \
    --max_sources N         \
    [--use_abstract]        \
    [--base_prompt PROMPT]  \
    [--seed N]
```

*All arguments in [] are optional.