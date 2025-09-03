import os, gzip
import pandas as pd

from typing import Any, Union
from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset
from datasets import Dataset

from .data_structures import Node, Graph
from .prompting import PromptGenerator


def generate_split_sets(split_idx: Any) -> dict[str, set[int]]:
    return {key: set(value.tolist()) for key, value in split_idx.items()}


def generate_graphs(data_path: str) -> tuple[Graph, Graph, Graph]:
    """Generate the TRAIN and TEST datasets' graph.

    Keyword arguments:
    - data_path: The path to the dataset files.
    """

    # Constants
    DATASET_NAME = "ogbn-arxiv"
    PAPER_INFORMATION_PATH = os.path.join(data_path, "titleabs.tsv")
    MAPPING_FILE_PATH = os.path.join(
        data_path, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz"
    )

    # Load datasets
    paper_information_df = pd.read_csv(PAPER_INFORMATION_PATH, sep="\t")
    paper_information_df.columns = ["id", "title", "abstract"]

    with gzip.open(MAPPING_FILE_PATH, "rt") as f:
        mapping_df = pd.read_csv(f)

    mapping_df.columns = ["node_id", "paper_id"]

    dataset_pyg = PygNodePropPredDataset(name=DATASET_NAME, root=data_path)
    graph_pyg = dataset_pyg[0]

    # Create merged dataset
    merged_df = paper_information_df.merge(
        mapping_df, how="inner", left_on="id", right_on="paper_id", suffixes=("", "_y")
    )

    merged_df = merged_df[["node_id", "id", "title", "abstract"]]

    merged_df["node_id"] = merged_df["node_id"].astype(int)
    merged_df["id"] = merged_df["id"].astype(int)
    merged_df["title"] = merged_df["title"].astype(str)
    merged_df["abstract"] = merged_df["abstract"].astype(str)

    merged_df.sort_values(by="node_id", inplace=True)

    # Extract edges and splits from graph
    citers = graph_pyg.edge_index[0].tolist()
    sources = graph_pyg.edge_index[1].tolist()

    split_idx = dataset_pyg.get_idx_split()
    split_sets = generate_split_sets(split_idx)

    # Create the graphs
    train_graph, validation_graph, test_graph = Graph(), Graph(), Graph()

    ## Add nodes
    NUMBER_OF_NODES = merged_df.shape[0]
    for row in tqdm(merged_df.itertuples(), total=NUMBER_OF_NODES, desc="Adding nodes"):
        node_id = row.node_id
        id = row.id
        title = row.title
        abstract = row.abstract

        node = Node(node_id=node_id, id=id, title=title, abstract=abstract)

        test_graph.add_node(node)

        if node_id not in split_sets["test"]:
            validation_graph.add_node(node)

        if node_id in split_sets["train"]:
            train_graph.add_node(node)

    ## Add edges
    NUMBER_OF_EDGES = len(sources)
    for src, dst in tqdm(
        zip(sources, citers), total=NUMBER_OF_EDGES, desc="Adding edges"
    ):
        SHOULD_ADD_TRAIN = (src in split_sets["train"]) and (dst in split_sets["train"])
        SHOULD_ADD_VALID = (
            src in split_sets["train"] or src in split_sets["valid"]
        ) and (dst in split_sets["valid"])
        SHOULD_ADD_TEST = dst in split_sets["test"]

        if SHOULD_ADD_TRAIN:
            train_graph.add_source(src, dst)

        elif SHOULD_ADD_VALID:
            validation_graph.add_source(src, dst)

        elif SHOULD_ADD_TEST:
            test_graph.add_source(src, dst)

    return train_graph, validation_graph, test_graph


def generate_dataset(
    graph: Graph,
    use_abstract: bool = False,
    max_sources: Union[int, None] = None,
    seed: int = 42,
    base_prompt: str | None = None,
) -> Dataset:
    """Generate a dataset from the graph.

    Keyword arguments:
    - graph: The input graph from which to generate the dataset.
    - use_abstract: Whether to use the abstract of the papers as input.
    - max_sources: The maximum number of source papers to consider.
    - seed: The random seed for reproducibility.
    - base_prompt: The base prompt to use for the questions.
    """

    prompter = PromptGenerator(graph, base_prompt=base_prompt)
    with_sources = graph.get_nodes_with_sources()

    dataset = dict(node_id=[], question=[], answer=[])

    for item in tqdm(with_sources, desc="Generating dataset"):
        dataset_item = prompter.generate_prompt(
            item,
            use_abstract=use_abstract,
            max_sources=max_sources,
            seed=seed,
        )

        dataset["node_id"].append(dataset_item["node_id"])
        dataset["question"].append(dataset_item["question"])
        dataset["answer"].append(dataset_item["answer"])

    return Dataset.from_dict(dataset)
