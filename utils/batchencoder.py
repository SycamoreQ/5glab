import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pickle
import os


class BatchEncoder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 256, 
        cache_file: str = "embeddings_cache.pkl"
    ):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.cache = {}
        
        # Load cache
        self._load_cache()
        
        print(f"BatchEncoder initialized")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Cached embeddings: {len(self.cache):,}")
    
    def _load_cache(self):
        """Load embedding cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache):,} cached embeddings")
            except:
                self.cache = {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def precompute_paper_embeddings(
        self,
        papers: List[Dict],
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        
        to_encode = []
        paper_ids = []
        texts = []
        
        for paper in papers:
            paper_id = paper['paper_id']
            
            if force or paper_id not in self.cache:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                fields = paper.get('fields', [])
                
                if abstract and len(abstract) > 50:
                    text = f"{title}. {abstract[:400]}" 
                elif fields and isinstance(fields, list) and len(fields) > 0:
                    field_str = ' '.join(str(f) for f in fields[:3])
                    text = f"{title}. {field_str}"
                else:
                    text = title
                
                texts.append(text)
                to_encode.append(text)
                paper_ids.append(paper_id)
        
        if not to_encode:
            print(" All embeddings cached!")
            return {p['paper_id']: self.cache[p['paper_id']] for p in papers}
        
        print(f"  Need to encode: {len(to_encode):,} papers")
        print(f"  Using device: {self.device}")
        
        # Batch encode on GPU
        print("  Encoding in batches...")
        embeddings = self.model.encode(
            to_encode,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        # Update cache
        for paper_id, embedding in zip(paper_ids, embeddings):
            self.cache[paper_id] = embedding
        
        # Save cache
        print("  Saving cache...")
        self._save_cache()
        
        print(f"✓ Precomputed {len(embeddings):,} new embeddings")
        
        return {p['paper_id']: self.cache[p['paper_id']] for p in papers}
    
    def encode_with_cache(
        self,
        text: str,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """Encode text with caching."""
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            device=self.device
        )
        
        if cache_key:
            self.cache[cache_key] = embedding
        
        return embedding
