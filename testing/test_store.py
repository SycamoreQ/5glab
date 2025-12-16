# test_edge_cache.py
import pickle
import asyncio
from graph.database.store import EnhancedStore

async def test():
    with open('training_cache/edge_cache_1M.pkl', 'rb') as f:
        edge_cache = pickle.load(f)
    
    with open('training_cache/training_papers_1M.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    # Pick a random paper
    paper = papers[0]
    pid = paper.get('paperId') or paper.get('paper_id')
    
    print(f"Testing paper: {pid}")
    print(f"Paper title: {paper.get('title')}")
    
    # Check edge cache
    edges = edge_cache.get(pid, [])
    print(f"Edges in cache: {len(edges)}")
    if edges:
        print(f"First 5 edges: {edges[:5]}")
    
    # Check if targets exist in Neo4j
    store = EnhancedStore()
    for edge_type, target_id in edges[:5]:
        target = await store.get_paper_by_id(target_id)
        if target:
            print(f"  ✓ Target {target_id[:10]}... exists in Neo4j")
        else:
            print(f"  ✗ Target {target_id[:10]}... NOT in Neo4j")
    
    await store.pool.close()

asyncio.run(test())
