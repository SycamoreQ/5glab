import asyncio
import pickle
import os
from collections import Counter
from graph.database.store import EnhancedStore


async def build_paper_cache_streaming(max_papers: int = None):
    print("Building paper cache with two-tier strategy\n")
    
    store = EnhancedStore(pool_size=10)
    
    # Recent papers with abstracts (2015+)
    query_with_abstracts = """
    MATCH (p:Paper)
    WHERE p.paperId IS NOT NULL 
      AND p.title IS NOT NULL
      AND p.abstract IS NOT NULL
      AND p.abstract <> ''
      AND p.year >= 2015
    WITH p
    OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
    WITH p, count(ref) as ref_count
    WHERE ref_count >= 2
    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
    WITH p, ref_count, count(citing) as cite_count
    WHERE cite_count >= 2
    RETURN 
        p.paperId as paper_id,
        p.title as title,
        p.year as year,
        p.abstract as abstract,
        p.fieldsOfStudy as fields,
        p.citationCount as citation_count,
        ref_count as referenceCount,
        cite_count as citationCount
    ORDER BY p.year DESC, (ref_count + cite_count) DESC
    SKIP $1
    LIMIT $2
    """
    
    # Highly-cited papers (any year, abstracts optional)
    query_highly_cited = """
    MATCH (p:Paper)
    WHERE p.paperId IS NOT NULL 
      AND p.title IS NOT NULL
    WITH p
    OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
    WITH p, count(ref) as ref_count
    WHERE ref_count >= 3
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
    
    # Phase 1: Get papers WITH abstracts (target: 70%)
    max_papers = max_papers or 10000
    target_with_abstracts = int(max_papers * 0.7)
    
    print(f"📚 Phase 1: Fetching papers WITH abstracts (target: {target_with_abstracts:,})")
    print(f"  Criteria: Year >= 2015, refs >= 2, cites >= 2\n")
    
    skip = 0
    phase1_attempts = 0
    max_attempts = 5
    
    while len(all_papers) < target_with_abstracts and phase1_attempts < max_attempts:
        try:
            results = await store._run_query_method(query_with_abstracts, [skip, batch_size])
        except Exception as e:
            print(f" Error at offset {skip:,}: {e}")
            break
        
        if not results:
            print(f"  Exhausted query at offset {skip:,}")
            break
        
        added_this_batch = 0
        for r in results:
            pid = r['paper_id']
            abstract = r.get('abstract', '')
            
            # Verify abstract is non-empty
            if pid not in seen_ids and abstract and len(abstract) > 50:
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
                added_this_batch += 1
        
        skip += batch_size
        phase1_attempts += 1
        
        if added_this_batch > 0:
            print(f"  Batch {phase1_attempts}: +{added_this_batch:,} papers | Total: {len(all_papers):,}")
        
        if len(all_papers) >= target_with_abstracts:
            print(f"  ✅ Reached target: {len(all_papers):,} papers with abstracts")
            break
        
        await asyncio.sleep(0.01)
    
    print(f"\n  Phase 1 complete: {len(all_papers):,} papers with abstracts\n")
    
    # Phase 2: Add highly-cited papers (fill remaining slots)
    remaining = max_papers - len(all_papers)
    
    if remaining > 0:
        print(f"🌟 Phase 2: Adding highly-cited papers (target: {remaining:,})")
        print(f"  Criteria: refs >= 3, cites >= 5, any year\n")
        
        skip = 0
        phase2_attempts = 0
        
        while len(all_papers) < max_papers and phase2_attempts < max_attempts:
            try:
                results = await store._run_query_method(query_highly_cited, [skip, batch_size])
            except Exception as e:
                print(f"  ⚠️ Error at offset {skip:,}: {e}")
                break
            
            if not results:
                print(f"  Exhausted query at offset {skip:,}")
                break
            
            added_this_batch = 0
            for r in results:
                pid = r['paper_id']
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append({
                        'paper_id': pid,
                        'title': r['title'],
                        'abstract': r.get('abstract', ''),
                        'year': r.get('year'),
                        'fields': r.get('fields', []),
                        'citation_count': r.get('citation_count', 0) or 0,
                        'referenceCount': r.get('referenceCount', 0)
                    })
                    added_this_batch += 1
                    
                    if len(all_papers) >= max_papers:
                        break
            
            skip += batch_size
            phase2_attempts += 1
            
            if added_this_batch > 0:
                print(f"  Batch {phase2_attempts}: +{added_this_batch:,} papers | Total: {len(all_papers):,}")
            
            if len(all_papers) >= max_papers:
                print(f"Reached max_papers: {len(all_papers):,}")
                break
            
            await asyncio.sleep(0.01)
    
    await store.pool.close()
    
    if not all_papers:
        print("\nNo papers found! Check your Neo4j data and connectivity.")
        return
    
    print(f"\n{'='*70}")
    print(f"CACHE STATISTICS")
    print(f"{'='*70}\n")
    
    # Abstract coverage
    papers_with_abstracts = [p for p in all_papers if p.get('abstract') and len(p['abstract']) > 50]
    papers_without_abstracts = [p for p in all_papers if not p.get('abstract') or len(p['abstract']) <= 50]
    
    print(f"Data Coverage:")
    print(f"  Total papers: {len(all_papers):,}")
    print(f"  With abstracts (>50 chars): {len(papers_with_abstracts):,} ({len(papers_with_abstracts)/len(all_papers)*100:.1f}%)")
    print(f"  Without abstracts: {len(papers_without_abstracts):,} ({len(papers_without_abstracts)/len(all_papers)*100:.1f}%)")
    
    # Year coverage
    papers_with_year = [p for p in all_papers if p.get('year')]
    papers_with_fields = [p for p in all_papers if p.get('fields')]
    
    print(f"\n  With year: {len(papers_with_year):,} ({len(papers_with_year)/len(all_papers)*100:.1f}%)")
    print(f"  With fields: {len(papers_with_fields):,} ({len(papers_with_fields)/len(all_papers)*100:.1f}%)")
    
    # Citation statistics
    papers_with_citations = [p for p in all_papers if p.get('citation_count', 0) > 0]
    if papers_with_citations:
        avg_cites = sum(p.get('citation_count', 0) for p in all_papers) / len(all_papers)
        max_cites = max(p.get('citation_count', 0) for p in all_papers)
        
        print(f"\nCitation Statistics:")
        print(f"  Papers with citations: {len(papers_with_citations):,}")
        print(f"  Avg citations: {avg_cites:.2f}")
        print(f"  Max citations: {max_cites:,}")
    
    # Year distribution
    if papers_with_year:
        year_dist = Counter(p['year'] for p in papers_with_year if p.get('year'))
        print(f"\nYear Distribution (Top 10):")
        for year in sorted(year_dist.keys(), reverse=True)[:10]:
            print(f"  {year}: {year_dist[year]:,} papers")
    
    # Field distribution
    field_counts = Counter()
    for p in all_papers:
        fields = p.get('fields', [])
        if isinstance(fields, list):
            for field in fields:
                if isinstance(field, str):
                    field_counts[field] += 1
    
    if field_counts:
        print(f"\nField Distribution (Top 10):")
        for field, count in field_counts.most_common(10):
            print(f"  {field}: {count:,} papers")
    
    # Top papers
    sorted_by_cites = sorted(all_papers, key=lambda p: p.get('citation_count', 0), reverse=True)
    print(f"\nTop 5 Most Cited Papers:")
    for i, p in enumerate(sorted_by_cites[:5], 1):
        title_preview = p['title'][:70] + "..." if len(p['title']) > 70 else p['title']
        has_abstract = "✅" if p.get('abstract') and len(p['abstract']) > 50 else ""
        print(f"  {i}. {title_preview}")
        print(f"     Citations: {p.get('citation_count', 0):,} | Year: {p.get('year', 'N/A')} | Abstract: {has_abstract}")
    
    # Sample abstracts
    if papers_with_abstracts:
        print(f"\nSample Abstract (first with abstract):")
        sample = papers_with_abstracts[0]
        abstract_preview = sample['abstract'][:200] + "..." if len(sample['abstract']) > 200 else sample['abstract']
        print(f"  Title: {sample['title'][:60]}")
        print(f"  Abstract: {abstract_preview}")
    
    # Save cache
    cache_file = 'training_papers.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(all_papers, f)
    
    file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"CACHE SAVED")
    print(f"{'='*70}")
    print(f"  File: {cache_file}")
    print(f"  Papers: {len(all_papers):,}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Abstract coverage: {len(papers_with_abstracts)/len(all_papers)*100:.1f}%")
    
    print(f"\nNEXT STEPS:")
    print(f"  1. Verify abstracts: python -c \"import pickle; papers = pickle.load(open('training_papers.pkl', 'rb')); print('Abstracts:', sum(1 for p in papers if len(p.get('abstract', '')) > 50))\"")
    print(f"  2. Run community detection: python -m graph.database.comm_det")
    print(f"  3. Train RL agent: python -m RL.train_rl --episodes 50")
    
    print(f"\nCache ready for training!")
    return all_papers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build paper cache for RL training (two-tier strategy)')
    parser.add_argument('--max-papers', type=int, default=10000,
                       help='Maximum papers to cache (default: 10000)')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"TWO-TIER PAPER CACHE BUILDER")
    print(f"{'='*70}")
    print(f"Target: {args.max_papers:,} papers")
    print(f"  - 70% recent papers with abstracts (2015+)")
    print(f"  - 30% highly-cited papers (any year)")
    print(f"{'='*70}\n")
    
    asyncio.run(build_paper_cache_streaming(max_papers=args.max_papers))
