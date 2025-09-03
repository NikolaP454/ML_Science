import os
import json, argparse

import dataset


def extract_arguments() -> argparse.Namespace:
    """Extract command line arguments."""

    argparser = argparse.ArgumentParser(
        description="Create data for training and testing."
    )

    argparser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the experiment files.",
    )

    argparser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the folder containing the data files (papers.csv, citations.csv, etc.).",
    )

    argparser.add_argument(
        "--use_abstract",
        action="store_true",
        help="Whether to use the abstract of the papers as input.",
    )

    argparser.add_argument(
        "--max_sources",
        type=int,
        required=True,
        help="The maximum number of source papers to consider.",
    )

    argparser.add_argument(
        "--base_prompt",
        type=str,
        default=None,
        help="The base prompt to use for the questions.",
    )

    argparser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )

    return argparser.parse_args()


def create_experiment_data():
    """Generate the experiment's data (train, validation and testing)."""

    # Extract Arguments
    args = extract_arguments()

    EXPERIMENT_PATH = args.experiment_path
    DATA_PATH_INPUT = args.data_path
    USE_ABSTRACT = args.use_abstract
    MAX_SOURCES = args.max_sources
    BASE_PROMPT = args.base_prompt
    SEED = args.seed

    # Path creation
    DATASET_PATH = os.path.join(EXPERIMENT_PATH, "datasets")
    CONFIG_PATH = os.path.join(EXPERIMENT_PATH, "config")

    os.makedirs(EXPERIMENT_PATH, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CONFIG_PATH, exist_ok=True)

    # Save Configs
    with open(os.path.join(CONFIG_PATH, "config_data_generation.json"), "w") as f:
        json.dump(
            {
                "input_data_path": DATA_PATH_INPUT,
                "use_abstract": USE_ABSTRACT,
                "max_sources": MAX_SOURCES,
                "base_prompt": BASE_PROMPT,
                "seed": SEED,
            },
            f,
        )

    # Create Datasets
    train_graph, validation_graph, testing_graph = dataset.generate_graphs(
        DATA_PATH_INPUT
    )

    train_ds = dataset.generate_dataset(
        train_graph,
        use_abstract=USE_ABSTRACT,
        max_sources=MAX_SOURCES,
        seed=SEED,
        base_prompt=BASE_PROMPT,
    )

    validation_ds = dataset.generate_dataset(
        validation_graph,
        use_abstract=USE_ABSTRACT,
        max_sources=MAX_SOURCES,
        seed=SEED,
        base_prompt=BASE_PROMPT,
    )

    testing_ds = dataset.generate_dataset(
        testing_graph,
        use_abstract=USE_ABSTRACT,
        max_sources=MAX_SOURCES,
        seed=SEED,
        base_prompt=BASE_PROMPT,
    )

    # Save Datasets
    train_ds.save_to_disk(os.path.join(DATASET_PATH, "train.jsonl"))
    validation_ds.save_to_disk(os.path.join(DATASET_PATH, "validation.jsonl"))
    testing_ds.save_to_disk(os.path.join(DATASET_PATH, "testing.jsonl"))


if __name__ == "__main__":
    create_experiment_data()
