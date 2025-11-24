import chromadb.proto
import ray
import io
from pathlib import Path
from typing import Dict, List , str , Any 
from unstructured.partition.auto import partition
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import uuid
from utils.config import BaseConfig
from chromadb.utils import embedding_functions
import paperscraper
import os 
import logging
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader


chroma_client = chromadb.PersistentClient(path = BaseConfig.chroma_path)
emb_fun = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name= BaseConfig.embedding_model
)


vector_collection = chroma_client.get_collection(
    name = "academic_papers",
    embedding_function=emb_fun,
    metadata ={"hnsw:space": "cosine"}
)


def fetch_and_vectorize_pdf(paper_data: Dict[str, Any]):
    """
    1. Downloads PDF via DOI.
    2. Chunks text.
    3. Ingests to ChromaDB.
    """
    doi = paper_data.get("doi")
    if not doi:
        logging.warning(f"No DOI for paper {paper_data['paper_id']}, skipping PDF fetch.")
        return

    # 1. Download PDF
    pdf_dir = "./temp_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{paper_data['paper_id']}.pdf")
    
    try:
        # Check if already processed
        existing = vector_collection.get(where={"paper_id": paper_data['paper_id']})
        if existing and len(existing['ids']) > 0:
            print(f"Vectors already exist for {doi}, skipping.")
            return

        print(f"Fetching PDF for DOI: {doi}...")
        # paperscraper tries arXiv, PubMed, etc.
        paperscraper.pdf.save_pdf({'doi': doi}, filepath=pdf_path)
        
        if not os.path.exists(pdf_path):
            logging.warning(f"Could not download PDF for {doi}")
            return

        # 2. Load & Chunk
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Academic-optimized splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
        )
        chunks = splitter.split_documents(documents)

        # 3. Vector Ingest
        ids = []
        metadatas = []
        texts = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"{paper_data['paper_id']}_{i}")
            texts.append(chunk.page_content)
            metadatas.append({
                "paper_id": paper_data['paper_id'], # CRITICAL LINK KEY
                "doi": doi,
                "title": paper_data['title'],
                "chunk_index": i
            })
            
        if texts:
            vector_collection.add(ids=ids, documents=texts, metadatas=metadatas)
            print(f" -> Ingested {len(texts)} chunks to ChromaDB.")
            
    except Exception as e:
        logging.error(f"Vectorization failed for {doi}: {e}")
    finally:
        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    


