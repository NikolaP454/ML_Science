import re
import numpy as np
from tqdm import tqdm

from typing import Union


def clean_text_data(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class Node:
    node_id: int
    id: int
    title: str
    abstract: str

    def __init__(self, node_id: int, id: int, title: str, abstract: str):
        self.node_id = int(node_id)
        self.id = int(id)
        self.title = clean_text_data(title)
        self.abstract = clean_text_data(abstract)

    def __repr__(self):
        return f"Node({self.node_id=}, {self.id=}, {self.title=}, {self.abstract=})"

    def __str__(self):
        return f"Node {self.node_id}: {self.title}"

    def get_prompt(
        self, *, use_abstract: bool = False, answer_tag: bool = False
    ) -> str:
        """Generate a prompt for the node.

        Keyword arguments:
        - use_abstract: Whether to include the abstract in the prompt.
        - answer_tag: Whether to use the "ANSWER" tag instead of "PAPER".
        """

        base_tag = "ANSWER" if answer_tag else "PAPER"
        prompt = f"<{base_tag}><TITLE>{self.title}</TITLE>"

        if use_abstract:
            prompt += f"<ABSTRACT>{self.abstract}</ABSTRACT>"

        prompt += f"</{base_tag}>"
        return prompt + "\n"


class Graph:
    nodes: dict[int, Node]
    sources: dict[int, list[int]]

    def __init__(self):
        self.nodes = {}
        self.sources = {}

    def __repr__(self) -> str:
        return f"Graph(num_nodes={len(self.nodes)}, with_sources={self.count_nodes_with_sources()})"

    def add_node(self, node: Node):
        self.nodes.update({node.node_id: node})
        self.sources.update({node.node_id: []})

    def create_and_add_node(self, node_id: int, id: int, title: str, abstract: str):
        node = Node(node_id=node_id, id=id, title=title, abstract=abstract)
        self.add_node(node)

    def add_source(self, source_id: int, dest_id: int):
        """Added a source for a given destination node.

        Keyword arguments:
        - source_id: The node ID of the source used by the destination node (paper).
        - dest_id: The node ID of the destination node (paper).
        """

        assert source_id in self.nodes, f"Source node {source_id} not in graph"
        assert dest_id in self.nodes, f"Destination node {dest_id} not in graph"

        if dest_id not in self.sources:
            self.sources[dest_id] = []

        self.sources[dest_id].append(source_id)

    def add_sources_from_lists(
        self, source_nodes: list[int], destination_nodes: list[int]
    ):
        """Added multiple edges from source and destination lists.

        Keyword arguments:
        - source_nodes: A list of source node IDs (papers).
        - destination_nodes: A list of destination node IDs (papers).
        """

        for source, dest in tqdm(
            zip(source_nodes, destination_nodes), total=len(source_nodes)
        ):
            self.add_source(source, dest)

    def count_nodes_with_sources(self) -> int:
        """Count the number of nodes with at least one source."""

        return sum(1 for sources in self.sources.values() if sources)

    def get_sources_for_node(
        self, node_id: int, max_count: Union[int, None] = None, seed: int = 42
    ) -> list[int]:
        """Get the source nodes for a given destination node.

        Keyword arguments:
        - node_id: The node ID of the destination node (paper).
        """

        assert node_id in self.nodes, f"Node {node_id} not in graph"
        assert max_count is None or max_count > 0, "max_count must be positive"
        assert (
            self.sources.get(node_id, None) is not None
        ), f"Node {node_id} has no sources"

        sources = self.sources.get(node_id, [])

        if max_count is not None:
            sources = (
                np.random.default_rng(seed)
                .choice(sources, size=min(max_count, len(sources)), replace=False)
                .tolist()
            )

        return sources

    def get_nodes_with_sources(self) -> list[int]:
        """Get a list of all node IDs that have at least one source."""

        return [node_id for node_id, sources in self.sources.items() if sources]

    def get_node(self, node_id: int) -> Node:
        """Get a node by its ID."""

        assert node_id in self.nodes, f"Node {node_id} not found"
        return self.nodes[node_id]
