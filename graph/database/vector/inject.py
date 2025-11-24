import ray
import io
from pathlib import Path
from typing import Dict, List
from unstructured.partition.auto import partition
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import uuid
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
)
from utils.config import BaseConfig


SOURCE_DIR = BaseConfig.database_path
ds = ray.data.read_binary_files(SOURCE_DIR, include_paths=True, concurrency=5)
ds.schema()



