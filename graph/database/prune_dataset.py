
import asyncio
import pickle
import numpy as np
from collections import Counter, defaultdict
from graph.database.store import EnhancedStore
from typing import Dict, List, Set, Tuple
from sentence_transformers import SentenceTransformer


async def extract_query_critical_papers(
    store, 
    queries: List[str],
    existing_embeddings: Dict,  
    encoder,
    top_k=100,
    min_year=2010
) -> List[Dict]:
    """Get papers highly relevant to queries, reusing existing embeddings."""
    
    print(f"\n{'='*70}")
    print("EXTRACTING QUERY-CRITICAL PAPERS (REUSING EMBEDDINGS)")
    print(f"{'='*70}")
    
    # Fetch candidates from Neo4j
    query = """
    MATCH (p:Paper)
    WHERE p.year >= $1
    AND p.paperId IS NOT NULL
    AND p.title IS NOT NULL
    AND p.abstract IS NOT NULL
    AND size(p.abstract) >= 20
    RETURN p.paperId as paperId,
           p.title as title,
           p.year as year,
           p.abstract as abstract,
           p.fieldsOfStudy as fields,
           p.citationCount as citationCount,
           p.venue as venue
    LIMIT 500000
    """
    
    candidates = await store._run_query_method(query, [min_year])
    print(f" Found {len(candidates):,} candidates from Neo4j")

    candidates_with_embeddings = [
        p for p in candidates 
        if str(p['paperId']) in existing_embeddings
    ]
    
    print(f"  ✓ {len(candidates_with_embeddings):,} already have embeddings")
    print(f"  ⚠ {len(candidates) - len(candidates_with_embeddings):,} would need embedding (skipping)")
    
    candidate_embeddings = {
        p['paperId']: existing_embeddings[str(p['paperId'])]
        for p in candidates_with_embeddings
    }
    
    query_papers = set()
    query_embeddings = encoder.encode(queries)
    
    for query, q_emb in zip(queries, query_embeddings):
        similarities = []
        for pid, p_emb in candidate_embeddings.items():
            sim = np.dot(q_emb, p_emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-9
            )
            similarities.append((sim, pid))
        
        similarities.sort(reverse=True)
        top_papers = [pid for _, pid in similarities[:top_k]]
        query_papers.update(top_papers)
        
        print(f"\n  Query: '{query[:50]}...'")
        print(f"    Top similarity: {similarities[0][0]:.3f}")
        print(f"    Added {len(top_papers)} papers")
    
    query_paper_list = [p for p in candidates_with_embeddings if p['paperId'] in query_papers]
    
    print(f"\n  ✓ Total query-critical papers: {len(query_paper_list):,}")
    return query_paper_list



async def extract_year_bucket(store, year_min: int, year_max: int,
                               min_abstract_len=50, min_degree=1,
                               target_count=50_000) -> List[Dict]:
    """Extract high-quality papers from a year bucket (UNCHANGED)."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING BUCKET: {year_min}-{year_max}")
    print(f"{'='*70}")
    
    query = """
    MATCH (p:Paper)
    WHERE p.year >= $1 AND p.year <= $2
    AND p.paperId IS NOT NULL
    AND p.title IS NOT NULL
    AND p.abstract IS NOT NULL
    AND size(p.abstract) >= $3
    WITH p
    OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
    WITH p, count(ref) as out_degree
    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
    WITH p, out_degree, count(citing) as in_degree
    WHERE (in_degree + out_degree) >= $4
    RETURN p.paperId as paperId,
           p.title as title,
           p.year as year,
           p.abstract as abstract,
           p.fieldsOfStudy as fields,
           p.citationCount as citationCount,
           p.venue as venue,
           in_degree,
           out_degree
    ORDER BY (in_degree + out_degree) DESC
    LIMIT $5
    """
    
    papers = await store._run_query_method(
        query,
        [year_min, year_max, min_abstract_len, min_degree, target_count]
    )
    
    if not papers:
        print(f"No papers found")
        return []
    
    print(f" ✓ Found {len(papers):,} papers")
    if papers:
        print(f" Avg degree: {np.mean([p['in_degree'] + p['out_degree'] for p in papers]):.1f}")
    return papers


async def expand_with_cited_papers(store, core_papers: List[Dict],
                                   max_refs_per_paper=20,
                                   include_citing_papers=True , max_citing_per_paper=10) -> Tuple[List[Dict], List[Dict]]:
    """Get citations (KEEP YOUR EXISTING CODE)."""
    print(f"\n{'='*70}")
    print("EXPANDING WITH CITED PAPERS")
    print(f"{'='*70}")
    
    core_ids = [p['paperId'] for p in core_papers]
    core_id_set = set(core_ids)
    cited_papers = {}
    all_edges = []
    
    citations_per_source = defaultdict(int)
    citations_per_target = defaultdict(int)
    
    batch_size = 5000
    total_batches = (len(core_ids) + batch_size - 1) // batch_size
    
    print(f" Fetching outgoing citations in {total_batches} batches...")
    print(f" Target: {max_refs_per_paper} citations per paper")
    
    for batch_idx, i in enumerate(range(0, len(core_ids), batch_size)):
        batch = core_ids[i:i+batch_size]
        
        citation_query = """
        UNWIND $1 AS pid
        MATCH (source:Paper {paperId: pid})-[:CITES]->(target:Paper)
        WHERE target.paperId IS NOT NULL
        RETURN source.paperId as source,
               target.paperId as target,
               target.title as target_title,
               target.year as target_year,
               target.abstract as target_abstract,
               target.fieldsOfStudy as target_fields,
               target.citationCount as target_citationCount,
               target.venue as target_venue
        """
        
        batch_citations = await store._run_query_method(citation_query, [batch])
        
        citations_by_source = defaultdict(list)
        for cit in batch_citations:
            citations_by_source[cit['source']].append(cit)
        
        for source_id, cits in citations_by_source.items():
            remaining = max_refs_per_paper - citations_per_source[source_id]
            if remaining > 0:
                for cit in cits[:remaining]:
                    target_id = cit['target']
                    all_edges.append({
                        'source': source_id,
                        'target': target_id
                    })
                    citations_per_source[source_id] += 1
                    
                    if target_id not in core_id_set and target_id not in cited_papers:
                        cited_papers[target_id] = {
                            'paperId': target_id,
                            'title': cit.get('target_title', 'Unknown'),
                            'year': cit.get('target_year'),
                            'abstract': cit.get('target_abstract'),
                            'fieldsOfStudy': cit.get('target_fields'),
                            'citationCount': cit.get('target_citationCount', 0),
                            'venue': cit.get('target_venue')
                        }
        
        if (batch_idx + 1) % 5 == 0:
            avg_so_far = len(all_edges) / len([s for s in citations_per_source if citations_per_source[s] > 0])
            print(f" Batch {batch_idx+1}/{total_batches}: {len(all_edges):,} total edges, avg {avg_so_far:.1f} per paper")
    
    print(f" ✓ Found {len(all_edges):,} outgoing citations")
    outgoing_count = len(all_edges)
    print(f"\n Fetching incoming citations...")
    
    for batch_idx, i in enumerate(range(0, len(core_ids), batch_size)):
        batch = core_ids[i:i+batch_size]
        
        citing_query = """
        UNWIND $1 AS pid
        MATCH (citing:Paper)-[:CITES]->(target:Paper {paperId: pid})
        WHERE citing.paperId IS NOT NULL
        RETURN citing.paperId as source,
               target.paperId as target,
               citing.title as source_title,
               citing.year as source_year,
               citing.abstract as source_abstract,
               citing.fieldsOfStudy as source_fields,
               citing.citationCount as source_citationCount,
               citing.venue as source_venue
        """
        
        batch_citing = await store._run_query_method(citing_query, [batch])
        
        citing_by_target = defaultdict(list)
        for cit in batch_citing:
            citing_by_target[cit['target']].append(cit)
        
        for target_id, cits in citing_by_target.items():
            remaining = max_citing_per_paper - citations_per_target[target_id]
            if remaining > 0:
                for cit in cits[:remaining]:
                    source_id = cit['source']
                    
                    # Add edge
                    all_edges.append({
                        'source': source_id,
                        'target': target_id
                    })
                    citations_per_target[target_id] += 1
                    
                    if source_id not in core_id_set and source_id not in cited_papers:
                        cited_papers[source_id] = {
                            'paperId': source_id,
                            'title': cit.get('source_title', 'Unknown'),
                            'year': cit.get('source_year'),
                            'abstract': cit.get('source_abstract'),
                            'fieldsOfStudy': cit.get('source_fields'),
                            'citationCount': cit.get('source_citationCount', 0),
                            'venue': cit.get('source_venue')
                        }
        
        if (batch_idx + 1) % 5 == 0:
            incoming = len(all_edges) - outgoing_count
            print(f" Batch {batch_idx+1}/{total_batches}: +{incoming:,} incoming edges")
    
    incoming_total = len(all_edges) - outgoing_count
    print(f" ✓ Added {incoming_total:,} incoming citations")
    print(f" ✓ Total edges: {len(all_edges):,}")
    print(f" ✓ Bridge papers: {len(cited_papers):,}")
    
    return list(cited_papers.values()), all_edges


async def main():
    store = EnhancedStore(pool_size=20)
    
    print("Loading encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading existing embeddings...")
    with open('training_cache/embeddings_1M.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings):,} existing embeddings\n")
    
    queries = [
        "deep learning convolutional neural networks",
        "natural language processing transformers",
        "reinforcement learning policy gradient",
        "ResNet deep residual learning image recognition",
        "BERT bidirectional encoder representations transformers",
        "proximal policy optimization PPO reinforcement learning",
        "attention is all you need vaswani transformer",
        "deep double Q-network DDQN prioritized replay",
        "AlexNet ImageNet convolutional neural network"
    ]

    query_critical = await extract_query_critical_papers(
        store, 
        queries, 
        existing_embeddings=embeddings,  
        encoder=encoder, 
        top_k=100, 
        min_year=2010
    )
    
    buckets = [
        (2018, 2019, 50_000),
        (2020, 2021, 50_000),
        (2022, 2023, 50_000),
        (2024, 2025, 50_000),
    ]
    
    bucket_papers = []
    for year_min, year_max, target in buckets:
        papers = await extract_year_bucket(
            store, year_min, year_max,
            min_abstract_len=150,
            min_degree=1,
            target_count=target
        )
        bucket_papers.extend(papers)
    
    seen_ids = set()
    core_papers = []
    for p in query_critical + bucket_papers:
        pid = p['paperId']
        if pid not in seen_ids:
            core_papers.append(p)
            seen_ids.add(pid)
    
    print(f"\n{'='*70}")
    print(f"CORE PAPERS: {len(core_papers):,}")
    print(f"  Query-critical: {len(query_critical):,}")
    print(f"  Bucket papers: {len(bucket_papers):,}")
    print(f"  After dedup: {len(core_papers):,}")
    print(f"{'='*70}")
    
    # 3. Expand with citations
    bridge_papers, citations = await expand_with_cited_papers(
        store, core_papers, max_refs_per_paper=20
    )
    
    all_papers = core_papers + bridge_papers
    paper_ids = [p['paperId'] for p in all_papers]
    
    graph_data = {
        'papers': all_papers,
        'citations': citations,
        'stats': {
            'num_papers': len(all_papers),
            'num_core_papers': len(core_papers),
            'num_bridge_papers': len(bridge_papers),
            'num_citations': len(citations),
            'query_critical_papers': len(query_critical)
        }
    }
    
    with open('pruned_graph_enhanced_1M.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    
    print(f"\n{'='*70}")
    print("✓ SAVED: pruned_graph_enhanced_1M.pkl")
    print(f"{'='*70}")
    print(f"Core papers: {len(core_papers):,}")
    print(f"Bridge papers: {len(bridge_papers):,}")
    print(f"Total papers: {len(all_papers):,}")
    print(f"Citations: {len(citations):,}")
    print(f"Query-critical: {len(query_critical):,}")
    
    await store.pool.close()
    return graph_data



if __name__ == '__main__':
    asyncio.run(main())
