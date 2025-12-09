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
        
        INVALID_VALUES = {'', 'N/A', '...', 'Unknown', 'null', 'None', 'undefined'}
        
        for paper in papers:
            paper_id = paper.get('paper_id')
            if not paper_id:
                continue
            
            title = str(paper.get('title', '')) if paper.get('title') else ''
            keywords = str(paper.get('fields' , '')) if paper.get('fieldOfStudy') else ''
            abstract = str(paper.get('abstract', '')) if paper.get('abstract') else ''
            pub_name = str(paper.get('venue', '')) if paper.get('venue') else ''
            
            if not title or title in INVALID_VALUES or len(title) <= 3:
                continue  
            
            parts = [title]
            
            if keywords:
                parts.append(keywords)
            
            if abstract and len(abstract) > 20:
                parts.append(abstract[:500])
            elif pub_name and pub_name not in INVALID_VALUES:
                parts.append(pub_name)
            
            text = " ".join(parts).strip()
            
            if text and len(text) >= 10:
                texts.append(text)
                paper_ids.append(paper_id)
        
        print(f"Precomputing embeddings for {len(texts)} papers...")
        
        if not texts:
            print("Warning: No valid texts to encode!")
            return {}

        print(f"\nSample texts being encoded:")
        for i, (text, pid) in enumerate(zip(texts[:3], paper_ids[:3]), 1):
            print(f"  {i}. {text[:80]}...")
            print(f"     ID: {pid}")
        
        embeddings = self.encode_batch(texts)
        
        valid_count = 0
        zero_count = 0
        
        embedding_map = {}
        for pid, emb in zip(paper_ids, embeddings):
            if isinstance(emb, np.ndarray) and emb.shape[0] > 0:
                if np.abs(emb).sum() > 0.01:  # Not all zeros
                    embedding_map[pid] = emb
                    valid_count += 1
                else:
                    zero_count += 1
            else:
                zero_count += 1
        
        print(f"Precomputed {len(embedding_map)} valid embeddings")
        if zero_count > 0:
            print(f"Warning: {zero_count} embeddings were zero/invalid")
        
        if embedding_map:
            sample_emb = next(iter(embedding_map.values()))
            print(f"Embedding dimension: {sample_emb.shape}")
            print(f"Sample embedding norm: {np.linalg.norm(sample_emb):.3f}")
        
        return embedding_map