import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore


async def build_paper_cache_streaming(max_papers: int = None):
    print("Building high-quality paper cache\n")
    
    store = EnhancedStore(pool_size=10)
    
    query = """
    MATCH (p:Paper)
    WHERE p.paperId IS NOT NULL 
      AND p.title IS NOT NULL
      AND p.abstract IS NOT NULL
      AND size(p.abstract) >= 200
      AND p.year >= 2018
    WITH p
    OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
    WITH p, count(ref) as ref_count
    WHERE ref_count >= 5
    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
    WITH p, ref_count, count(citing) as cite_count
    WHERE cite_count >= 5
    RETURN 
        p.paperId as paper_id,
        p.title as title,
        p.year as year,
        p.abstract as abstract,
        p.fieldsOfStudy as fields,
        p.citationCount as citation_count,
        ref_count as referenceCount,
        cite_count as citationCount
    ORDER BY (ref_count + cite_count) DESC
    SKIP $1
    LIMIT $2
    """
    
    batch_size = 5000
    all_papers = []
    seen_ids = set()
    
    max_papers = max_papers or 3000
    
    print(f"Target: {max_papers:,} high-quality papers")
    print(f"Criteria: Year >= 2018, abstract >= 200 chars, refs/cites >= 5\n")
    
    skip = 0
    attempts = 0
    max_attempts = 3
    
    while len(all_papers) < max_papers and attempts < max_attempts:
        try:
            results = await store._run_query_method(query, [skip, batch_size])
        except Exception as e:
            print(f"Error: {e}")
            break
        
        if not results:
            break
        
        added = 0
        for r in results:
            pid = r['paper_id']
            abstract = r.get('abstract', '')
            
            if pid not in seen_ids and len(abstract) >= 200:
                seen_ids.add(pid)
                all_papers.append({
                    'paper_id': pid,
                    'title': r['title'],
                    'abstract': abstract,
                    'year': r.get('year'),
                    'fields': r.get('fields', []),
                    'citation_count': r.get('citation_count', 0) or 0,
                    'referenceCount': r.get('referenceCount', 0)
                })
                added += 1
        
        skip += batch_size
        attempts += 1
        
        if added > 0:
            print(f"Batch {attempts}: +{added:,} papers | Total: {len(all_papers):,}")
        
        if len(all_papers) >= max_papers:
            break
        
        await asyncio.sleep(0.01)
    
    await store.pool.close()
    
    if not all_papers:
        print("\nNo papers found!")
        return
    
    print(f"\n{'='*70}")
    print(f"CACHE STATISTICS")
    print(f"{'='*70}\n")
    
    papers_with_abstracts = [p for p in all_papers if len(p.get('abstract', '')) >= 200]
    
    print(f"Total papers: {len(all_papers):,}")
    print(f"With abstracts (≥200 chars): {len(papers_with_abstracts):,} ({len(papers_with_abstracts)/len(all_papers)*100:.1f}%)")
    
    avg_cites = sum(p.get('citation_count', 0) for p in all_papers) / len(all_papers)
    avg_refs = sum(p.get('referenceCount', 0) for p in all_papers) / len(all_papers)
    
    print(f"Avg citations: {avg_cites:.2f}")
    print(f"Avg references: {avg_refs:.2f}")
    
    year_dist = Counter(p['year'] for p in all_papers if p.get('year'))
    print(f"\nYear Distribution:")
    for year in sorted(year_dist.keys(), reverse=True)[:5]:
        print(f"  {year}: {year_dist[year]:,} papers")
    
    cache_file = 'training_papers.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(all_papers, f)
    
    file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"CACHE SAVED")
    print(f"{'='*70}")
    print(f"File: {cache_file}")
    print(f"Papers: {len(all_papers):,}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Abstract coverage: {len(papers_with_abstracts)/len(all_papers)*100:.1f}%")
    
    return all_papers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-papers', type=int, default=3000)
    args = parser.parse_args()
    
    asyncio.run(build_paper_cache_streaming(max_papers=args.max_papers))
