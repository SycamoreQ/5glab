import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore

async def build_author_cache(max_authors: int = 1000):
    """
    Build a cache of high-quality, influential authors.
    Memory-efficient version using simpler queries.
    """
    print("Building high-quality author cache\n")
    store = EnhancedStore(pool_size=10)
    
    # Simpler query that uses existing properties instead of complex aggregations
    query = """
    MATCH (a:Author)-[:WROTE]->(p:Paper)
    WHERE a.authorId IS NOT NULL
      AND a.name IS NOT NULL
    WITH a, COUNT(DISTINCT p) as paper_count
    WHERE paper_count >= 5
    RETURN 
      a.authorId as author_id,
      a.name as name,
      paper_count
    ORDER BY paper_count DESC
    LIMIT $1
    """
    
    print(f"Target: {max_authors:,} high-quality authors")
    print(f"Criteria: Papers >= 5, Active researchers\n")
    
    try:
        results = await store._run_query_method(query, [max_authors])
        print(f"Retrieved {len(results):,} authors from database")
    except Exception as e:
        print(f"Error: {e}")
        await store.pool.close()
        return
    
    if not results:
        print("\nNo authors found!")
        await store.pool.close()
        return
    
    # Enrich authors with additional metadata (one at a time to avoid memory issues)
    all_authors = []
    print("Enriching author data...")
    
    for i, r in enumerate(results):
        author_id = r['author_id']
        
        try:
            # Get papers by this author
            papers = await store.get_papers_by_author(author_id)
            
            # Calculate metrics from papers
            total_citations = sum(p.get('citationcount', 0) or 0 for p in papers)
            recent_papers = sum(1 for p in papers if p.get('year', 0) and p['year'] >= 2020)
            
            # Get collaborators count
            collabs = await store.get_collabs_by_author(author_id)
            collaborator_count = len(collabs)
            
            # Get h-index
            h_index = await store.get_author_h_index(author_id)
            
            # Get citation velocity
            citation_velocity = await store.get_citation_velocity(author_id)
            
            # Get publication diversity
            pub_diversity = await store.get_pub_diversity(author_id)
            
            # Calculate impact score
            impact_score = (
                r['paper_count'] * 2.0 + 
                total_citations / 10.0 + 
                collaborator_count + 
                h_index * 5.0
            )
            
            author_data = {
                'author_id': author_id,
                'name': r['name'],
                'paper_count': r['paper_count'],
                'total_citations': total_citations,
                'collaborator_count': collaborator_count,
                'recent_papers': recent_papers,
                'impact_score': impact_score,
                'h_index': h_index,
                'citation_velocity': citation_velocity,
                'pub_diversity': pub_diversity
            }
            
            all_authors.append(author_data)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(results)} authors...")
                
        except Exception as e:
            print(f"Error processing author {author_id}: {e}")
            continue
    
    await store.pool.close()
    
    if not all_authors:
        print("\nNo authors could be processed!")
        return
    
    # Sort by impact score
    all_authors.sort(key=lambda x: x['impact_score'], reverse=True)
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"AUTHOR CACHE STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Total authors: {len(all_authors):,}")
    
    avg_papers = sum(a['paper_count'] for a in all_authors) / len(all_authors)
    avg_citations = sum(a['total_citations'] for a in all_authors) / len(all_authors)
    avg_collabs = sum(a['collaborator_count'] for a in all_authors) / len(all_authors)
    avg_hindex = sum(a['h_index'] for a in all_authors) / len(all_authors)
    
    print(f"Avg papers: {avg_papers:.1f}")
    print(f"Avg citations: {avg_citations:.1f}")
    print(f"Avg collaborators: {avg_collabs:.1f}")
    print(f"Avg h-index: {avg_hindex:.1f}")
    
    # Top authors
    print(f"\nTop 10 Authors by Impact:")
    for i, author in enumerate(all_authors[:10]):
        print(f"  {i+1}. {author['name'][:50]}")
        print(f"      h-index: {author['h_index']}, papers: {author['paper_count']}, citations: {author['total_citations']}")
    
    # Paper count distribution
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
    
    # Save cache
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
    
    return all_authors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build author training cache')
    parser.add_argument('--max-authors', type=int, default=1000, 
                       help='Maximum number of authors to cache (default: 1000)')
    args = parser.parse_args()
    
    asyncio.run(build_author_cache(max_authors=args.max_authors))
