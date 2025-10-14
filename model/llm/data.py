from graph.database.store import EnhancedStore
from pipeline import Pipeline
import json 
import huggingface_hub 
from pipeline import Pipeline
from dataclasses import dataclass
from typing import List, Dict, Any
import os
import logging
import dspy
from utils.config import BaseConfig
from datasets import Dataset, DatasetDict, load_dataset
import kuzu 
from tqdm import tqdm
from model.llm.prompt import optimize_pipeline_with_mipro
import pandas as pd


class TrainingDataset: 

    def __init__(self , is_config = True):
        self.is_config = is_config 
        self.config = BaseConfig() if is_config else None
        self.pipeline = Pipeline()
        self.store = EnhancedStore()
        self.db_path = self.config.kuzu_db_path if is_config else "research_db"
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)

    def preprocess(self , data_dir: str , instruction_key , response_key):
        
        dataset = load_dataset("json" , data_dir= BaseConfig.data_path , split="train")
        
        formatted = []

        for sample in dataset:
            instr = sample.get(instruction_key)
            resp = sample.get(response_key)
            
            if instr and resp:
                formatted.append({
                    "instruction": instr,
                    "response": resp
                })

        return formatted
    
    def load_arxiv(self):
        if self.is_config and self.config.use_arxiv:
            logging.info("Loading arXiv dataset...")
            arxiv_data = self.preprocess(
                data_dir=self.config.arxiv_data_path,
                instruction_key="question",
                response_key="answer"
            )
            return arxiv_data
        return []
    
    def load_sciqa(self):
        if self.is_config and self.config.use_sciqa:
            logging.info("Loading SciQA dataset...")
            sciqa_data = self.preprocess(
                data_dir=self.config.sciqa_data_path,
                instruction_key="question",
                response_key="answer"
            )
            return sciqa_data
        return []
    

    def get_papers_with_authors_and_citations(self , limit=200 ):
        """Retrieve paper-author-citation triples from Kuzu."""
        query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        OPTIONAL MATCH (p)-[:CITED]->(cited:Paper)
        RETURN a.name AS author, p.title AS paper, p.abstract AS abstract,
                collect(DISTINCT cited.title) AS citations
        LIMIT $1
        """
        result = self.conn.execute(query, [limit])
        records = []
        for row in result:
            records.append({
                "author": row[0],
                "paper": row[1],
                "abstract": row[2],
                "citations": [c for c in row[3] if c]
            })
        return records

    def generate_academic_qa(record):
        """Turn a record into one or more QA pairs."""
        author = record["author"]
        paper = record["paper"]
        abstract = record["abstract"] or "No abstract available."
        citations = record["citations"]

        # Construct context from graph info
        context = f"The paper '{paper}' authored by {author} discusses: {abstract}"
        if citations:
            context += f" It has been cited by {len(citations)} other research papers such as {', '.join(citations[:3])}."

        # Generate multiple QA pairs per record
        questions_and_answers = [
            {
                "question": f"What are the main findings of the paper '{paper}' by {author}?",
                "answer": abstract.split(".")[0] + "."
            },
            {
                "question": f"How has the paper '{paper}' influenced other works?",
                "answer": f"It has been cited by {len(citations)} other papers, indicating its influence in the research community."
                if citations else "It has not been cited widely yet."
            },
            {
                "question": f"Who authored the paper '{paper}'?",
                "answer": f"The paper '{paper}' was authored by {author}."
            }
        ]

        dataset = []
        for qa in questions_and_answers:
            dataset.append({
                "question": qa["question"],
                "context": context,
                "answer": qa["answer"]
            })
        return dataset


    
        

    