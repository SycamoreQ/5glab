from typing import Dict, List  , Any 
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile

chroma_path = tempfile.mkdtemp(prefix="chroma_test_")
print("Using temp chroma path:", chroma_path)
config = BaseConfig()


chroma_client = chromadb.PersistentClient(path = chroma_path)
emb_fun = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name= 'all-MiniLM-L6-v2'
)


vector_collection = chroma_client.get_or_create_collection(
    name="academic_papers",
    embedding_function=emb_fun
)



def fetch_and_vectorize_pdf(paper_data: Dict[str, Any]):
    """
    1. Downloads PDF via DOI.
    2. Chunks text.
    3. Ingests to ChromaDB.
    """
    doi = paper_data.get("doi")
    if not doi:
        logging.warning(f"No DOI for paper {paper_data.get('paper_id', 'UNKNOWN')}, skipping PDF fetch.")
        return

    pdf_dir = "./temp_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{paper_data['paper_id']}.pdf")
    
    try:
        existing = vector_collection.get(where={"paper_id": paper_data['paper_id']})
        if existing and len(existing['ids']) > 0:
            print(f"Vectors already exist for {doi}, skipping.")
            return

        print(f"Fetching PDF for DOI: {doi}...")
        paperscraper.pdf.save_pdf({'doi': doi}, filepath=pdf_path)
        
        if not os.path.exists(pdf_path):
            logging.warning(f"Could not download PDF for {doi}")
            return

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
        )
        chunks = splitter.split_documents(documents)

        ids = []
        metadatas = []
        texts = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"{paper_data['paper_id']}_{i}")
            texts.append(chunk.page_content)
            metadatas.append({
                "paper_id": paper_data['paper_id'], 
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
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    


