import asyncio
import pickle
import random
from graph.database.store import EnhancedStore


async def build_training_cache():
    """Cache high-quality papers using memory-efficient streaming."""
    print("Building training paper cache...")
    
    store = EnhancedStore(pool_size=20)
    
    print("Phase 1: Sampling papers with connectivity...")
    
    batch_size = 50000
    max_batches = 100
    all_papers = []
    
    for batch_num in range(max_batches):
        offset = batch_num * batch_size
        
        query = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL 
          AND p.title <> ''
          AND size(p.title) > 10
        WITH p
        SKIP $1
        LIMIT $2
        
        OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        
        WITH p, 
             count(DISTINCT ref) as ref_count, 
             count(DISTINCT citing) as cite_count
        WHERE ref_count > 0 OR cite_count > 0
        
        RETURN elementId(p) as paper_id,
               p.title as title,
               p.year as year,
               COALESCE(p.doi, p.id, '') as doi,
               p.publication_name,
               p.keywords,
               p.id as original_id,
               ref_count,
               cite_count
        """
        
        try:
            batch = await store._run_query_method(query, [offset, batch_size])
            
            if not batch:
                print(f"  Batch {batch_num + 1}: No more papers")
                break
            
            all_papers.extend(batch)
            print(f"  Batch {batch_num + 1}: Found {len(batch)} papers (total: {len(all_papers)})")
            
            if len(all_papers) >= 200000:
                print(f"  Reached 50k papers, stopping...")
                break
                
        except Exception as e:
            print(f"  Batch {batch_num + 1}: Error - {e}")
            break
    
    print(f"\nPhase 2: Filtering quality papers...")
    
    valid_papers = []
    for p in all_papers:
        title = p.get('title', '')
        ref_count = p.get('ref_count', 0)
        cite_count = p.get('cite_count', 0)
        
        if (title and 
            len(title) > 10 and
            title not in ['...', 'research paper', ''] and
            (ref_count + cite_count) >= 3):
            valid_papers.append(p)
    
    print(f"  Valid papers: {len(valid_papers)}")
    
    if not valid_papers:
        print("\nERROR: No valid papers found!")
        await store.pool.close()
        return
    
    print("\nPhase 3: Sorting by connectivity...")
    valid_papers.sort(key=lambda x: x['ref_count'] + x['cite_count'], reverse=True)
    
    cached_papers = valid_papers[:20000]
    
    with open('training_papers.pkl', 'wb') as f:
        pickle.dump(cached_papers, f)
    
    print(f"\n✓ Cache saved to training_papers.pkl")
    
    print("\nTop 10 most connected papers:")
    for i, p in enumerate(cached_papers[:10], 1):
        total_conn = p['ref_count'] + p['cite_count']
        print(f"  {i}. {p['title'][:60]}")
        print(f"     Refs: {p['ref_count']}, Cites: {p['cite_count']}, Total: {total_conn}")
    
    total_refs = sum(p['ref_count'] for p in cached_papers)
    total_cites = sum(p['cite_count'] for p in cached_papers)
    avg_refs = total_refs / len(cached_papers)
    avg_cites = total_cites / len(cached_papers)
    
    print(f"\nCache statistics:")
    print(f"  Total papers: {len(cached_papers)}")
    print(f"  Avg references: {avg_refs:.1f}")
    print(f"  Avg citations: {avg_cites:.1f}")
    print(f"  Min connectivity: {min(p['ref_count'] + p['cite_count'] for p in cached_papers)}")
    print(f"  Max connectivity: {max(p['ref_count'] + p['cite_count'] for p in cached_papers)}")
    
    await store.pool.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(build_training_cache())
