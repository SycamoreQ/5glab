import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)


class BaseConfig:
    
    llm_name: str = field(
        default = "meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Class name indicating which LLM model to use."}
    )

    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )

    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )

    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )

    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )

    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )

    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )

    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )

    data_path_arxiv: str = field(
        default = "taesiri/arxiv_qa",
        metadata = {"help" : "Path to the dataset in huggingface for fine tuning LLM"}
    )

    data_path_sciqa: str = field(
        default = "orkg/SciQA",
        metadata = {"help" : "Path to the dataset in huggingface for fine tuning LMM"}
    )

    database_path: str = field(
        default="data/kuzu/5g_graph",
        metadata={"help": "Path to the Kuzu graph database."}
    )

    output_path: str = field(
        default = "data/kuzu_generated_data.jsonl",
        metadata = {"help" : "Path to save the custom generated data from Kuzu for fine tuning"}
    )

    use_arxiv: bool = field(
        default=True,
        metadata={"help": "Whether to use the arXiv dataset for training."}
    )

    use_sciqa: bool = field(
        default=True,
        metadata={"help": "Whether to use the SciQA dataset for training."}
    )

    

    