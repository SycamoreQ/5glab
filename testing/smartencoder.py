"""
Optimal encoding strategy based on test results:
- Use title-only as baseline
- Only add abstract when it improves similarity
- Use smart weighted when we do add abstract
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re


class OptimalEncoder:
    """
    Encoding strategy optimized for your dataset.
    
    Key insight from tests:
    - Title-only works best for specific queries (0.385 vs 0.339)
    - Smart weighted helps for broad queries (0.545 vs 0.530)
    - Naive concat always hurts (0.339 vs 0.385)
    
    Solution: Use title with optional smart abstract weighting
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Very light abstract weight (title dominates)
        self.title_weight = 0.85  # Increased from 0.7
        self.abstract_weight = 0.15  # Decreased from 0.3
    
    def encode_paper_optimal(self, title: str, abstract: str = None, 
                            use_abstract: bool = True) -> np.ndarray:
        """
        Optimal encoding that preserves title signal.
        
        Strategy:
        1. Always encode title (strongest signal)
        2. If abstract exists and is high-quality, add it with LOW weight
        3. Only use first 2 sentences of abstract (most specific part)
        """
        # Always encode title
        title_emb = self.model.encode(title)
        
        # Return title-only if no abstract or disabled
        if not use_abstract or not abstract or len(abstract) < 200:
            return title_emb
        
        # Extract only first 2 sentences (most specific)
        first_sentences = self._get_first_n_sentences(abstract, n=2)
        
        if len(first_sentences) < 50:
            # Abstract too short after extraction
            return title_emb
        
        # Encode abstract portion
        abstract_emb = self.model.encode(first_sentences)
        
        # Very light weighting - title dominates
        combined = (self.title_weight * title_emb + 
                   self.abstract_weight * abstract_emb)
        
        # Normalize
        combined = combined / (np.linalg.norm(combined) + 1e-9)
        
        return combined
    
    def _get_first_n_sentences(self, text: str, n: int = 2) -> str:
        """Extract first N sentences."""
        sentences = re.split(r'[.!?]+', text)
        first_n = sentences[:n]
        result = ". ".join(s.strip() for s in first_n if s.strip())
        return result + "." if result else ""
    
    def encode_batch_optimal(self, papers: List[Dict], 
                           use_abstract: bool = True) -> Dict[str, np.ndarray]:
        """
        Batch encode papers with optimal strategy.
        
        Args:
            papers: List of paper dicts
            use_abstract: Whether to use abstracts at all (can disable for testing)
        """
        embeddings = {}
        
        for paper in papers:
            paper_id = paper.get('paper_id')
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if not title or not paper_id:
                continue
            
            emb = self.encode_paper_optimal(title, abstract, use_abstract)
            embeddings[paper_id] = emb
        
        return embeddings


# Update for batchencoder.py
def precompute_paper_embeddings_optimal(papers: List[Dict], 
                                       encoder_model = None,
                                       use_abstracts: bool = True) -> Dict[str, np.ndarray]:
    """
    Optimized version of precompute_paper_embeddings.
    
    Use this in batchencoder.py instead of current implementation.
    """
    if encoder_model is None:
        encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = []
    paper_ids = []
    
    INVALID_VALUES = {'', 'N/A', '...', 'Unknown', 'null', 'None', 'undefined'}
    
    title_weight = 0.85
    abstract_weight = 0.15
    
    print(f"Precomputing embeddings for {len(papers)} papers...")
    print(f"  Strategy: Title-dominant (85% title, 15% abstract first 2 sentences)")
    
    # First pass: encode all titles
    title_embeddings = {}
    for paper in papers:
        paper_id = paper.get('paper_id')
        title = str(paper.get('title', '')) if paper.get('title') else ''
        
        if not paper_id or not title or title in INVALID_VALUES or len(title) <= 3:
            continue
        
        texts.append(title)
        paper_ids.append(paper_id)
    
    print(f"  Encoding {len(texts)} titles...")
    title_embs = encoder_model.encode(texts, batch_size=32, show_progress_bar=True)
    
    for pid, emb in zip(paper_ids, title_embs):
        title_embeddings[pid] = emb
    
    # Second pass: add abstract if available
    if use_abstracts:
        print(f"  Adding abstract context (light weighting)...")
        
        for paper in papers:
            paper_id = paper.get('paper_id')
            abstract = paper.get('abstract', '')
            
            if paper_id not in title_embeddings:
                continue
            
            if not abstract or len(abstract) < 200:
                continue
            
            # Extract first 2 sentences only
            sentences = re.split(r'[.!?]+', abstract)
            first_two = ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."
            
            if len(first_two) < 50:
                continue
            
            # Encode abstract portion
            abstract_emb = encoder_model.encode(first_two)
            
            # Combine with light abstract weight
            title_emb = title_embeddings[paper_id]
            combined = (title_weight * title_emb + abstract_weight * abstract_emb)
            combined = combined / (np.linalg.norm(combined) + 1e-9)
            
            title_embeddings[paper_id] = combined
    
    print(f"✓ Precomputed {len(title_embeddings)} embeddings")
    
    return title_embeddings


def compare_strategies():
    """
    Compare title-only vs optimal strategy on your dataset.
    """
    import pickle
    
    print("="*80)
    print("FINAL COMPARISON: Title-Only vs Optimal Strategy")
    print("="*80)
    
    with open('training_papers_enriched.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    with_abstract = [p for p in papers if p.get('abstract') and len(p['abstract']) > 100][:100]
    
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    test_queries = [
        "machine learning neural networks",
        "deep learning computer vision", 
        "natural language processing",
        "reinforcement learning",
        "computer vision image recognition"
    ]
    
    title_only_scores = []
    optimal_scores = []
    
    print(f"\nTesting on {len(with_abstract)} papers...\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        query_emb = encoder.encode(query)
        
        keywords = query.lower().split()
        matching = []
        for paper in with_abstract:
            title_lower = paper['title'].lower()
            if any(kw in title_lower for kw in keywords):
                matching.append(paper)
                if len(matching) >= 5:
                    break
        
        if not matching:
            print("  No matches\n")
            continue
        
        for paper in matching[:2]:
            title = paper['title']
            abstract = paper['abstract']
            
            # Title only
            title_emb = encoder.encode(title)
            sim_title = np.dot(query_emb, title_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(title_emb)
            )
            
            # Optimal strategy
            optimal_encoder = OptimalEncoder()
            optimal_emb = optimal_encoder.encode_paper_optimal(title, abstract)
            sim_optimal = np.dot(query_emb, optimal_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(optimal_emb)
            )
            
            title_only_scores.append(sim_title)
            optimal_scores.append(sim_optimal)
            
            print(f"  {title[:60]}...")
            print(f"    Title-only: {sim_title:.3f}")
            print(f"    Optimal:    {sim_optimal:.3f} {'✓' if sim_optimal >= sim_title else '✗'}")
        
        print()
    
    # Statistics
    avg_title = np.mean(title_only_scores)
    avg_optimal = np.mean(optimal_scores)
    
    print("="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Average similarity:")
    print(f"  Title-only: {avg_title:.3f}")
    print(f"  Optimal:    {avg_optimal:.3f}")
    print(f"  Difference: {avg_optimal - avg_title:+.3f}")
    
    if avg_optimal > avg_title + 0.01:
        print(f"\n✓ OPTIMAL strategy wins!")
        print(f"  Use the optimal encoder in production")
    elif avg_optimal < avg_title - 0.01:
        print(f"\n⚠ Title-only is still better")
        print(f"  Stick with title-only encoding")
    else:
        print(f"\n≈ Both strategies are equivalent")
        print(f"  Use title-only for simplicity")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR YOUR SYSTEM:")
    print("="*80)
    
    if avg_title < 0.35:
        print("\n❌ OVERALL SIMILARITY TOO LOW (<0.35)")
        print("\nRoot cause: Your Kaggle dataset has generic/old papers")
        print("  The highly-cited 'classic' papers have generic titles")
        print("  Example: 'A Comparative Study...' doesn't match 'machine learning'")
        print("\nBEST SOLUTION: Download arXiv dataset")
        print("  • 2M papers with specific, modern titles")
        print("  • Full abstracts (not truncated)")
        print("  • Expected similarity: 0.5-0.7")
        print("  • Download: kaggle datasets download -d Cornell-University/arxiv")
        print("\nALTERNATIVE: Adjust your reward system")
        print("  • Lower similarity thresholds (0.3 instead of 0.7)")
        print("  • Increase structural rewards (citations, communities)")
        print("  • This is workable but suboptimal")
    else:
        print(f"\n✓ Similarity is acceptable ({avg_title:.3f})")
        print(f"  Update batchencoder.py with optimal strategy")


if __name__ == "__main__":
    compare_strategies()