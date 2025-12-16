"""
Pre-compute embeddings for the training cache (150K papers).
Works with the new 3-pickle structure:
  - training_cache/training_papers_1M.pkl
  - training_cache/edge_cache_1M.pkl
  - training_cache/paper_id_set_1M.pkl
"""
import pickle
import numpy as np
from utils.batchencoder import BatchEncoder
import asyncio
from graph.database.store import EnhancedStore

def cache_embeddings():
    """Pre-compute embeddings with connectivity validation."""
    print("="*80)
    print("CACHING EMBEDDINGS FOR TRAINING")
    print("="*80)
    
    # Load papers from new cache structure
    print("\nLoading training cache...")
    with open('training_cache/training_papers_1M.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    with open('training_cache/edge_cache_1M.pkl', 'rb') as f:
        edge_cache = pickle.load(f)
    
    with open('training_cache/paper_id_set_1M.pkl', 'rb') as f:
        paper_id_set = pickle.load(f)
    
    print(f"✓ Loaded {len(papers):,} papers")
    print(f"✓ Loaded {len(edge_cache):,} nodes in edge cache")
    print(f"✓ Loaded {len(paper_id_set):,} paper IDs")
    
    # Initialize encoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'
    )
    
    # Embed papers
    print("\n=== Embedding Papers ===")
    paper_embeddings = encoder.precompute_paper_embeddings(papers, force=False)
    
    print(f"\n✓ Total embeddings: {len(encoder.cache):,}")
    
    # Build validated query pools
    queries = [
        "deep learning convolutional neural networks",
        "natural language processing transformers",
        "reinforcement learning policy gradient",
        "graph neural networks message passing",
        "computer vision object detection"
    ]
    
    print("\n=== Building Query Pools ===")
    query_pools = build_query_pools(papers, encoder, edge_cache, queries)
    
    # Save query pools
    with open('training_cache/query_pools_1M.pkl', 'wb') as f:
        pickle.dump(query_pools, f)
    
    print(f"\n✓ Saved query pools to training_cache/query_pools_1M.pkl")
    print("\n" + "="*80)
    print("CACHING COMPLETE")
    print("="*80)

def build_query_pools(papers, encoder, edge_cache, queries, top_k=200):
    """Build query pools using connectivity from edge cache."""
    print(f"Building pools for {len(queries)} queries...")
    
    query_pools = {}
    
    for query in queries:
        print(f"\n Query: '{query[:40]}'")
        
        # Encode query
        query_emb = encoder.encode_with_cache(query, cache_key=f"query_{query}")
        
        # Score all papers
        paper_scores = []
        for paper in papers:
            pid = paper.get('paperId') or paper.get('paper_id')
            if pid and pid in encoder.cache:
                paper_emb = encoder.cache[pid]
                sim = np.dot(query_emb, paper_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
                )
                
                # Check connectivity from edge cache
                neighbors = edge_cache.get(pid, [])
                degree = len(neighbors)
                
                paper_scores.append((sim, paper, pid, degree))
        
        # Sort by similarity
        paper_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Validate connectivity (prefer papers with degree >= 3)
        validated = []
        for sim, paper, pid, degree in paper_scores[:top_k*2]:  # Check 2x to get top_k valid
            if degree >= 3:  # At least 3 neighbors
                validated.append((sim, paper, degree))
            if len(validated) >= top_k:
                break
        
        # If not enough well-connected papers, add lower-degree ones
        if len(validated) < top_k:
            for sim, paper, pid, degree in paper_scores:
                if len(validated) >= top_k:
                    break
                if (sim, paper, degree) not in validated and degree > 0:
                    validated.append((sim, paper, degree))
        
        query_pools[query] = [p for _, p, _ in validated]
        
        avg_sim = np.mean([s for s, _, _ in validated]) if validated else 0
        avg_degree = np.mean([d for _, _, d in validated]) if validated else 0
        
        print(f"   ✓ {len(validated)} papers | sim={avg_sim:.3f} | avg_degree={avg_degree:.1f}")
    
    return query_pools

if __name__ == '__main__':
    cache_embeddings()
