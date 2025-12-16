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
        cache_file: str = "training_cache/embeddings_1M.pkl"  # ← Updated path
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
                print(f"✓ Loaded {len(self.cache):,} cached embeddings")
            except Exception as e:
                print(f"⚠ Error loading cache: {e}")
                self.cache = {}
        else:
            print(f"No cache found at {self.cache_file}")
    
    def _save_cache(self):
        """Save embedding cache."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"✓ Saved {len(self.cache):,} embeddings to {self.cache_file}")
    
    def precompute_paper_embeddings(
        self,
        papers: List[Dict],
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Precompute embeddings for papers.
        Handles both 'paperId' and 'paper_id' key formats.
        """
        to_encode = []
        paper_ids = []
        texts = []
        
        print(f"Processing {len(papers):,} papers...")
        
        for paper in papers:
            # Handle both key formats
            paper_id = paper.get('paperId') or paper.get('paper_id')
            if not paper_id:
                continue
            
            if force or paper_id not in self.cache:
                title = paper.get('title', '') or ''
                abstract = paper.get('abstract', '') or ''
                fields = paper.get('fieldsOfStudy') or paper.get('fields', [])
                
                # Build text representation
                if abstract and len(abstract) > 50:
                    text = f"{title}. {abstract[:400]}"
                elif fields and isinstance(fields, list) and len(fields) > 0:
                    field_str = ' '.join(str(f) for f in fields[:3])
                    text = f"{title}. {field_str}"
                elif title and len(title) > 5:
                    text = title
                else:
                    text = f"Research paper {paper_id[-10:]}"
                
                texts.append(text)
                to_encode.append(text)
                paper_ids.append(paper_id)
        
        if not to_encode:
            print("  ✓ All embeddings already cached!")
            return {
                (p.get('paperId') or p.get('paper_id')): self.cache[p.get('paperId') or p.get('paper_id')] 
                for p in papers 
                if (p.get('paperId') or p.get('paper_id')) in self.cache
            }
        
        print(f"  Encoding {len(to_encode):,} papers on {self.device}...")
        
        # Batch encode
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
        self._save_cache()
        
        print(f"✓ Precomputed {len(embeddings):,} new embeddings")
        print(f"✓ Total cache size: {len(self.cache):,}")
        
        return {
            (p.get('paperId') or p.get('paper_id')): self.cache.get(p.get('paperId') or p.get('paper_id'))
            for p in papers
            if (p.get('paperId') or p.get('paper_id')) in self.cache
        }
    
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
    
    def encode(self, text, **kwargs):
        """Compatibility method for sentence_transformers API."""
        if isinstance(text, str):
            return self.encode_with_cache(text)
        elif isinstance(text, list):
            return self.model.encode(text, device=self.device, **kwargs)
        else:
            return self.model.encode(text, device=self.device, **kwargs)
