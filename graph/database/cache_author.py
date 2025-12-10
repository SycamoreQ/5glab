import asyncio
import pickle 
from graph.database.store import EnhancedStore

async def build_author_cache(): 

    store = EnhancedStore(pool_size=20)

    batch_size = 50000
    max_batches = 50 
    all_authors = []
    target_authors = 500000

    for batch_num in range(max_batches): 
        offset = batch_num * batch_size


        query = """
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            WITH a, count(DISTINCT p) as paper_count
            WHERE paper_count >= 3  // At least 3 papers
            
            OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
            WHERE a <> collab
            
            OPTIONAL MATCH (a)-[:WROTE]->(p2:Paper)<-[:CITES]-(citing:Paper)
            
            WITH a,
                paper_count,
                count(DISTINCT collab) as collab_count,
                count(DISTINCT citing) as total_citations
            
            RETURN a.authorId as author_id,
                a.name as name,
                paper_count,
                collab_count,
                total_citations,
                (paper_count * 2 + collab_count + total_citations / 10.0) as influence_score
            ORDER BY influence_score DESC
            LIMIT 50000
            """
        
    try: 
        all_authors = await store._run_query_method(query , [])
        print(f"found {len(all_authors)} Authors")

    except Exception as e: 
        print(f"  Error: {e}")
        await store.pool.close()
        return
    
    if not all_authors:
        print("\nERROR: No authors found!")
        await store.pool.close()
        return
    
    print("\nPhase 2: Filtering quality authors...")
    
    # Filter for active researchers
    valid_authors = [
        a for a in all_authors
        if a.get('paper_count', 0) >= 3 and a.get('name')
    ]
    
    print(f"  Valid authors: {len(valid_authors)}")
    
    # Take top 10k most influential
    cached_authors = valid_authors[:10000]

    with open('training_authors.pkl' , 'wb') as f: 
        pickle.dump(cached_authors , f)

    print(f"\n Cache saved to training_authors.pkl")
    
    print("\nTop 10 most influential authors:")
    for i, a in enumerate(cached_authors[:10], 1):
        print(f"  {i}. {a['name']}")
        print(f"     Papers: {a['paper_count']}, Collabs: {a['collab_count']}, Citations: {a['total_citations']}")
        print(f"     Influence: {a['influence_score']:.1f}")
    
    total_papers = sum(a['paper_count'] for a in cached_authors)
    total_collabs = sum(a['collab_count'] for a in cached_authors)
    total_cites = sum(a['total_citations'] for a in cached_authors)
    
    print(f"\nCache statistics:")
    print(f"  Total authors: {len(cached_authors):,}")
    print(f"  Avg papers: {total_papers / len(cached_authors):.1f}")
    print(f"  Avg collaborators: {total_collabs / len(cached_authors):.1f}")
    print(f"  Avg citations: {total_cites / len(cached_authors):.1f}")
    print(f"  Min papers: {min(a['paper_count'] for a in cached_authors)}")
    print(f"  Max papers: {max(a['paper_count'] for a in cached_authors)}")
    
    await store.pool.close()
    print("\n Done!")

if __name__ == "__main__":
    asyncio.run(build_author_cache())
