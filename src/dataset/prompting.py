from typing import Union

from .data_structures import Graph


class PromptGenerator:
    graph: Graph
    base_prompt: str

    def __init__(self, graph: Graph, base_prompt: str | None = None):
        self.graph = graph

        if base_prompt is None:
            base_prompt = "You are a senior researcher."

        self.base_prompt = base_prompt

    def __repr__(self) -> str:
        return f"PromptGenerator(graph={self.graph})"

    def generate_prompt(
        self,
        node_id: int,
        *,
        use_abstract: bool = False,
        max_sources: Union[int, None] = None,
        seed: int = 42,
    ) -> dict[str, Union[int, str]]:
        """Generates a prompt based on the graph and the provided parameters.

        Keyword arguments:
        - node_id: the ID of the node to generate a prompt for
        - use_abstract: whether to use abstract concepts in the prompt
        - max_sources: the maximum number of sources to include in the prompt
        - seed: the random seed for reproducibility
        """

        node = self.graph.get_node(node_id)
        sources = self.graph.get_sources_for_node(
            node_id, max_count=max_sources, seed=seed
        )

        question = f"{self.base_prompt} Given the following papers:".strip() + "\n"

        for source in sources:
            question += self.graph.get_node(source).get_prompt(
                use_abstract=use_abstract
            )

        question += "\nGenerate a new paper based on these sources. Return the paper in the following format:\n"
        question += "<ANSWER><TITLE>YOUR ANSWER HERE</TITLE></ANSWER>"

        answer = node.get_prompt(answer_tag=True)

        return {"node_id": node_id, "question": question, "answer": answer}
