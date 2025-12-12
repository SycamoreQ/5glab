import pickle
import numpy as np
from utils.batchencoder import BatchEncoder

def cache_embeddings():
    """Pre-compute embeddings with connectivity validation."""
    print("Loading caches...")
    
    with open('training_papers.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    with open('training_authors.pkl', 'rb') as f:
        authors = pickle.load(f)
    
    print(f"Loaded {len(papers):,} papers and {len(authors):,} authors\n")
    
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='embeddings_cache.pkl'
    )
    
    print("\n=== Embedding Papers ===")
    paper_embeddings = encoder.precompute_paper_embeddings(papers, force=False)
    
    print("\n=== Embedding Authors ===")
    author_texts = []
    author_ids = []
    
    for author in authors:
        aid = author.get('author_id')
        if aid and aid not in encoder.cache:
            name = author.get('name', 'Unknown')
            paper_count = author.get('paper_count', 0)
            h_index = author.get('h_index', 0)
            text = f"{name}, {paper_count} papers, h-index {h_index}"
            author_texts.append(text)
            author_ids.append(aid)
    
    if author_texts:
        print(f"  Encoding {len(author_texts):,} authors...")
        author_embeddings = encoder.model.encode(
            author_texts,
            batch_size=encoder.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=encoder.device
        )
        
        for aid, emb in zip(author_ids, author_embeddings):
            encoder.cache[aid] = emb
        
        encoder._save_cache()
        print(f"✓ Cached {len(author_embeddings):,} author embeddings")
    
    print(f"\n✓ Total embeddings: {len(encoder.cache):,}")
    
    # Build validated query pools
    import asyncio
    from graph.database.store import EnhancedStore
    
    queries = [
        "attention mechanism transformers",
        "graph neural networks",
        "reinforcement learning optimization",
        "computer vision deep learning",
        "natural language processing"
    ]
    
    print("\n=== Building Validated Query Pools ===")
    
    async def validate():
        store = EnhancedStore()
        query_pools = {}
        
        for query in queries:
            query_emb = encoder.encode_with_cache(query, cache_key=f"query_{query}")
            
            paper_scores = []
            for paper in papers:
                pid = paper.get('paper_id')
                if pid in encoder.cache:
                    paper_emb = encoder.cache[pid]
                    sim = np.dot(query_emb, paper_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
                    )
                    paper_scores.append((sim, paper, pid))
            
            paper_scores.sort(reverse=True, key=lambda x: x[0])
            
            print(f"\n  Query: '{query[:40]}'")
            print(f"  Validating connectivity...")
            
            validated = []
            for sim, paper, pid in paper_scores[:200]:
                refs = await store.get_references_by_paper(pid)
                cites = await store.get_citations_by_paper(pid)
                authors_list = await store.get_authors_by_paperid(pid)
                
                total_neighbors = len(refs or []) + len(cites or []) + len(authors_list or [])
                
                if total_neighbors >= 5 and (refs or cites):
                    validated.append((sim, paper, total_neighbors))
                
                if len(validated) >= 50:
                    break
            
            pool = {
                'papers': [p for _, p, _ in validated],
                'authors': authors[:50]
            }
            
            query_pools[query] = pool
            avg_sim = np.mean([s for s, _, _ in validated])
            avg_neighbors = np.mean([n for _, _, n in validated])
            print(f"    ✓ {len(validated)} papers | sim={avg_sim:.3f} | neighbors={avg_neighbors:.1f}")
        
        return query_pools
    
    query_pools = asyncio.run(validate())
    
    with open('query_pools_cache.pkl', 'wb') as f:
        pickle.dump(query_pools, f)
    
    print(f"\n✓ Saved query pools")

if __name__ == '__main__':
    cache_embeddings()
