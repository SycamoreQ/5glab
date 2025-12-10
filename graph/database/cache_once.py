import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore


async def build_full_abstract_cache(database: str = "neo4j"):
    """
    Get ALL papers with abstracts.
    This is your effective database for RL training.
    """
    print("="*80)
    print("BUILDING CACHE: ALL PAPERS WITH ABSTRACTS")
    print("="*80)
    print(f"Database: {database}\n")
    
    store = EnhancedStore(pool_size=20)
    
    # Get ALL papers with abstracts
    query = """
    MATCH (p:Paper)
    WHERE p.abstract IS NOT NULL 
      AND size(p.abstract) > 50
    
    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
    WHERE citing.abstract IS NOT NULL
    
    OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
    WHERE cited.abstract IS NOT NULL
    
    RETURN 
        p.paperId as paper_id,
        p.title as title,
        p.abstract as abstract,
        p.year as year,
        p.doi as doi,
        p.fieldsOfStudy as fields,
        p.citationCount as total_citations,
        count(DISTINCT citing) as cite_count,
        count(DISTINCT cited) as ref_count
    ORDER BY p.citationCount DESC
    """
    
    print("Querying ALL papers with abstracts...")
    
    results = await store._run_query_method(query, [])
    
    if not results:
        print(" No papers found!")
        await store.pool.close()
        return
    
    papers = []
    for r in results:
        papers.append({
            'paper_id': r['paper_id'],
            'title': r['title'],
            'abstract': r['abstract'],
            'year': r['year'],
            'doi': r.get('doi'),
            'arxiv_id': None,
            'venue': None,
            'fields': r.get('fields', []),
            'cite_count': r['cite_count'],
            'ref_count': r['ref_count'],
            'total_citations': r.get('total_citations', 0),
            'total_connectivity': r['cite_count'] + r['ref_count']
        })
    
    print(f"✓ Found {len(papers):,} papers with abstracts")
    
    # Check internal connectivity (only among papers WITH abstracts)
    print(f"\n🔍 Checking citation connectivity...")
    
    paper_ids = [p['paper_id'] for p in papers]
    
    edge_query = """
    MATCH (p1:Paper)-[:CITES]->(p2:Paper)
    WHERE p1.paperId IN $1 
      AND p2.paperId IN $1
      AND p1.abstract IS NOT NULL
      AND p2.abstract IS NOT NULL
    RETURN count(*) as edge_count
    """
    
    edge_result = await store._run_query_method(edge_query, [paper_ids])
    
    await store.pool.close()
    
    edge_count = 0
    if edge_result:
        edge_count = edge_result[0]['edge_count']
        nodes = len(papers)
        avg_degree = (2 * edge_count / nodes) if nodes > 0 else 0
        
        print(f"  Papers: {nodes:,}")
        print(f"  Internal citation edges: {edge_count:,}")
        print(f"  Avg degree: {avg_degree:.1f}")
        
        if avg_degree >= 10:
            print(f"  ✅ EXCELLENT! Use Leiden communities")
            strategy = "leiden"
        elif avg_degree >= 5:
            print(f"  ✅ Good! Use Leiden communities")
            strategy = "leiden"
        elif avg_degree >= 2:
            print(f"  ⚠️  Moderate. Use tier-based communities")
            strategy = "tier"
        else:
            print(f"  ⚠️  Sparse. Use tier-based communities")
            strategy = "tier"
    
    # Statistics
    avg_cites = sum(p['cite_count'] for p in papers) / len(papers)
    avg_refs = sum(p['ref_count'] for p in papers) / len(papers)
    avg_total = sum(p['total_citations'] for p in papers) / len(papers)
    
    print(f"\nCitation Statistics:")
    print(f"  Avg internal citations: {avg_cites:.1f}")
    print(f"  Avg internal references: {avg_refs:.1f}")
    print(f"  Avg total citations (S2ORC): {avg_total:.1f}")
    print(f"  Internal coverage: {(avg_cites + avg_refs) / avg_total * 100:.1f}%")
    
    # Year distribution
    year_dist = Counter(p['year'] for p in papers if p['year'])
    print(f"\nYear Distribution:")
    for year in sorted(year_dist.keys(), reverse=True)[:10]:
        print(f"  {year}: {year_dist[year]:,} papers")
    
    # Field distribution
    field_counts = Counter()
    for p in papers:
        fields = p.get('fields', [])
        if isinstance(fields, list):
            for field in fields[:2]:
                if isinstance(field, str):
                    field_counts[field] += 1
    
    print(f"\nField Distribution (Top 10):")
    for field, count in field_counts.most_common(10):
        print(f"  {field}: {count:,} papers")
    
    # Top papers
    print(f"\nTop 5 Most Connected Papers:")
    sorted_papers = sorted(papers, key=lambda p: p['total_connectivity'], reverse=True)
    for i, p in enumerate(sorted_papers[:5], 1):
        print(f"  {i}. {p['title'][:60]}...")
        print(f"     Internal: {p['cite_count']} cites, {p['ref_count']} refs")
        print(f"     Year: {p['year']}")
    
    # Save
    cache_file = 'training_papers_filtered.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(papers, f)
    
    file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
    
    print(f"\n✓ Saved to: {cache_file}")
    print(f"  Papers: {len(papers):,}")
    print(f"  Internal edges: {edge_count:,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Next steps
    print(f"\n📋 NEXT STEPS:")
    if strategy == "leiden":
        print(f"  1. Run: python -m graph.database.comm_det --leiden")
    else:
        print(f"  1. Run: python -m graph.database.comm_det  # Tier-based")
    
    print(f"  2. Train: python -m RL.train_rl --parser llm --episodes 100")
    print(f"\n  Your {len(papers):,} papers with {edge_count:,} edges is a GREAT dataset!")
    print(f"  Training will be fast and results should be excellent! 🚀")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='neo4j')
    
    args = parser.parse_args()
    
    asyncio.run(build_full_abstract_cache(database=args.database))
