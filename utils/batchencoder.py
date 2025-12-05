import torch 
from sentence_transformers import SentenceTransformer
from typing import Dict , List , Any 
import numpy as np 



class BatchEncoder:
    
    def __init__(self , model_name: str = "all-MiniLM-L6-v2" , device:str = "cuda"): 
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache = {}
        self.cache_limit = 10000

    
    def encode_batch(self , texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts , 
            batch_size = 32, 
            show_progress_bar= False ,
            convert_to_numpy= True , 
            device = self.device
        )

        return embeddings
    
    def encode_with_cache(self , text : str , cache_keys: str) -> np.ndarray: 
        key = cache_keys or text[:100]

        if key in self.cache: 
            return self.cache[key]
        
        embedding = self.model.encode(text , convert_to_numpy=True)

        if len(self.cache) < self.cache_limit: 
            self.cache[key] = embedding

        return embedding
    

    def precompute_paper_embeddings(self, papers: List[Dict]) -> Dict[str, np.ndarray]:
        texts = []
        paper_ids = []
        
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            text = f"{title} {abstract}".strip()
            
            if text:
                texts.append(text)
                paper_ids.append(paper['paper_id'])
        
        print(f"Precomputing embeddings for {len(texts)} papers...")
        embeddings = self.encode_batch(texts)
        
        embedding_map = {pid: emb for pid, emb in zip(paper_ids, embeddings)}
        print(f"✓ Precomputed {len(embedding_map)} embeddings")
        
        return embedding_map