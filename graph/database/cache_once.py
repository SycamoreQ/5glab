import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore

async def build_paper_cache_simple(max_papers: int = 3000):
    print("Building high-quality paper cache\n")
    store = EnhancedStore(pool_size=10)
    
    query = """
    MATCH (p:Paper)
    WHERE p.paperId IS NOT NULL
      AND p.title IS NOT NULL
      AND p.abstract IS NOT NULL
      AND size(p.abstract) >= 200
      AND p.year >= 2015
      AND p.citationCount >= 5
      AND p.referenceCount >= 5
    RETURN
      p.paperId as paper_id,
      p.title as title,
      p.year as year,
      p.abstract as abstract,
      p.fieldsOfStudy as fields,
      p.citationCount as citation_count,
      p.referenceCount as referenceCount
    ORDER BY p.citationCount DESC
    LIMIT $1
    """
    
    print(f"Target: {max_papers:,} high-quality papers")
    print(f"Criteria: Year >= 2018, abstract >= 200 chars, citationCount/referenceCount >= 5\n")
    
    try:
        # Single query with LIMIT
        results = await store._run_query_method(query, [max_papers])
        print(f"Retrieved {len(results):,} papers from database")
    except Exception as e:
        print(f"Error: {e}")
        await store.pool.close()
        return
    
    # Process results
    all_papers = []
    for r in results:
        all_papers.append({
            'paper_id': r['paper_id'],
            'title': r['title'],
            'abstract': r['abstract'],
            'year': r.get('year'),
            'fields': r.get('fields', []),
            'citation_count': r.get('citation_count', 0) or 0,
            'referenceCount': r.get('referenceCount', 0)
        })
    
    await store.pool.close()
    
    if not all_papers:
        print("\nNo papers found!")
        return
    
    # Statistics
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
    for year in sorted(year_dist.keys(), reverse=True):
        print(f"  {year}: {year_dist[year]:,} papers")
    
    # Save cache
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
    
    asyncio.run(build_paper_cache_simple(max_papers=args.max_papers))
