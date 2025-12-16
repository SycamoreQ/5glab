import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore


async def build_author_cache_small_batches(max_authors: int = 10000, batch_size: int = 1000):
    print("Building high-quality author cache (MEMORY-EFFICIENT VERSION)\n")
    store = EnhancedStore(pool_size=5)
    
    print(f"Target: {max_authors:,} high-quality authors")
    print(f"Criteria: Papers >= 5\n")
    print("Step 1: Finding qualifying authors...")
    
    id_query = """
    MATCH (a:Author)-[:WROTE]->(p:Paper)
    WHERE a.authorId IS NOT NULL
      AND a.name IS NOT NULL
    WITH a, COUNT(DISTINCT p) as paper_count
    WHERE paper_count >= 5
    RETURN a.authorId as author_id, a.name as name, paper_count
    ORDER BY paper_count DESC
    LIMIT $1
    """
    
    try:
        author_ids = await store._run_query_method(id_query, [max_authors])
        print(f"✓ Found {len(author_ids):,} qualifying authors\n")
    except Exception as e:
        print(f"Error: {e}")
        await store.pool.close()
        return
    
    if not author_ids:
        print("No authors found!")
        await store.pool.close()
        return
    
    print(f"Step 2: Enriching authors in batches of {batch_size:,}...")
    all_authors = []
    
    total_batches = (len(author_ids) + batch_size - 1) // batch_size
    
    for batch_num in range(0, len(author_ids), batch_size):
        batch = [a['author_id'] for a in author_ids[batch_num:batch_num+batch_size]]
        
        query1 = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        WHERE a.authorId IN $1
          AND p.year IS NOT NULL
        
        WITH a, 
             COUNT(DISTINCT p) as paper_count,
             SUM(COALESCE(p.citationCount, 0)) as total_citations,
             SIZE([year IN COLLECT(DISTINCT p.year) WHERE year >= 2020]) as recent_papers
        
        RETURN 
          a.authorId as author_id,
          a.name as name,
          paper_count,
          total_citations,
          recent_papers
        """
    
        query2 = """
        MATCH (a:Author)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
        WHERE a.authorId IN $1
          AND a <> collab
        
        WITH a, COUNT(DISTINCT collab) as collab_count
        
        RETURN a.authorId as author_id, collab_count
        """
        
        try:
            results1 = await store._run_query_method(query1, [batch])
            results2 = await store._run_query_method(query2, [batch])
        
            collab_map = {r['author_id']: r['collab_count'] for r in results2}
            
            for r in results1:
                author_id = r['author_id']
                paper_count = r['paper_count']
                total_citations = r['total_citations']
                collab_count = collab_map.get(author_id, 0)
                
                h_index = min(paper_count, int((total_citations / max(paper_count, 1)) ** 0.5))
                
                impact_score = (
                    paper_count * 2.0 + 
                    total_citations / 10.0 + 
                    collab_count + 
                    h_index * 5.0
                )
                
                all_authors.append({
                    'author_id': author_id,
                    'name': r['name'],
                    'paper_count': paper_count,
                    'total_citations': total_citations,
                    'collaborator_count': collab_count,
                    'recent_papers': r['recent_papers'],
                    'impact_score': impact_score,
                    'h_index': h_index,
                    'citation_velocity': 0,  #skip for speed (will add later)
                    'pub_diversity': 0       #skip for speed (will add later)
                })
            
            processed = min(batch_num + batch_size, len(author_ids))
            current_batch = (batch_num // batch_size) + 1
            print(f"  Batch {current_batch}/{total_batches}: Processed {processed:,}/{len(author_ids):,} authors")
            
        except Exception as e:
            print(f"Error processing batch {current_batch}: {e}")
            print(f"  Trying to continue...")
            continue
    
    await store.pool.close()
    
    if not all_authors:
        print("\nNo authors could be processed!")
        return
    
    all_authors.sort(key=lambda x: x['impact_score'], reverse=True)
    
    print(f"AUTHOR CACHE STATISTICS")
    
    print(f"Total authors: {len(all_authors):,}")
    
    if all_authors:
        avg_papers = sum(a['paper_count'] for a in all_authors) / len(all_authors)
        avg_citations = sum(a['total_citations'] for a in all_authors) / len(all_authors)
        avg_collabs = sum(a['collaborator_count'] for a in all_authors) / len(all_authors)
        avg_hindex = sum(a['h_index'] for a in all_authors) / len(all_authors)
        
        print(f"Avg papers: {avg_papers:.1f}")
        print(f"Avg citations: {avg_citations:.1f}")
        print(f"Avg collaborators: {avg_collabs:.1f}")
        print(f"Avg h-index: {avg_hindex:.1f}")
    
    if len(all_authors) >= 10:
        print(f"\nTop 10 Authors by Impact:")
        for i, author in enumerate(all_authors[:10]):
            print(f"  {i+1}. {author['name'][:50]}")
            print(f"      h-index: {author['h_index']}, papers: {author['paper_count']}, citations: {author['total_citations']}")
    
    paper_bins = Counter()
    for a in all_authors:
        count = a['paper_count']
        if count < 10:
            paper_bins['5-9'] += 1
        elif count < 20:
            paper_bins['10-19'] += 1
        elif count < 50:
            paper_bins['20-49'] += 1
        else:
            paper_bins['50+'] += 1
    
    print(f"\nPaper Count Distribution:")
    for bin_name in ['5-9', '10-19', '20-49', '50+']:
        if bin_name in paper_bins:
            print(f"  {bin_name} papers: {paper_bins[bin_name]:,} authors")
    
    cache_file = 'training_authors.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(all_authors, f)
    
    file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"AUTHOR CACHE SAVED")
    print(f"{'='*70}")
    print(f"File: {cache_file}")
    print(f"Authors: {len(all_authors):,}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Estimated time: {total_batches * 3} seconds ({total_batches} batches × 3 sec/batch)")
    
    return all_authors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build author cache (memory-efficient)')
    parser.add_argument('--max-authors', type=int, default=50000, 
                       help='Maximum number of authors to cache (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size (smaller = slower but safer, default: 1000)')
    args = parser.parse_args()
    
    asyncio.run(build_author_cache_small_batches(
        max_authors=args.max_authors,
        batch_size=args.batch_size
    ))
