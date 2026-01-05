import pickle
import os
from collections import defaultdict


def normalize_id(id_str: str) -> str:
    """Normalize ID to consistent format."""
    if not id_str:
        return ""
    id_str = str(id_str).strip()
    id_str = id_str.lstrip('0')
    return id_str if id_str else "0"


def main():
    print("EXTRACTING TO TRAINING CACHE FORMAT")
    input_file = 'pruned_graph_1M.pkl'
    print(f"\nLoading {input_file}...")
    
    with open(input_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    papers = graph_data['papers']
    citations = graph_data['citations']
    
    print(f"✓ Loaded {len(papers):,} papers")
    print(f"✓ Loaded {len(citations):,} citations")
    
    cache_dir = 'training_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\nSaving papers...")
    papers_file = os.path.join(cache_dir, 'training_papers_1M.pkl')
    with open(papers_file, 'wb') as f:
        pickle.dump(papers, f)
    
    file_size = os.path.getsize(papers_file) / 1024 / 1024
    print(f"Saved {papers_file} ({file_size:.1f} MB)")
    
    papers_with_authors = sum(1 for p in papers if p.get('authors'))
    avg_authors = sum(len(p.get('authors', [])) for p in papers) / len(papers)
    print(f"  Papers with authors: {papers_with_authors:,} ({100*papers_with_authors/len(papers):.1f}%)")
    print(f"  Avg authors per paper: {avg_authors:.1f}")
    
    print(f"\nBuilding edge cache...")
    edge_cache = defaultdict(list)
    
    for edge in citations:
        source = normalize_id(str(edge['source']))
        target = normalize_id(str(edge['target']))
        
        edge_cache[source].append(('cites', target))
        edge_cache[target].append(('citedby', source))
    
    edge_cache = dict(edge_cache)
    
    print(f"✓ Built edge cache with {len(edge_cache):,} nodes")
    print(f"  Total directed edges: {sum(len(v) for v in edge_cache.values()):,}")
    
    edges_file = os.path.join(cache_dir, 'edge_cache_1M.pkl')
    with open(edges_file, 'wb') as f:
        pickle.dump(edge_cache, f)
    
    file_size = os.path.getsize(edges_file) / 1024 / 1024
    print(f"✓ Saved {edges_file} ({file_size:.1f} MB)")
    
    # 3. Build paper ID set
    print(f"\nBuilding paper ID set...")
    paper_id_set = {normalize_id(str(p['paperId'])) for p in papers}
    
    paper_ids_file = os.path.join(cache_dir, 'paper_id_set_1M.pkl')
    with open(paper_ids_file, 'wb') as f:
        pickle.dump(paper_id_set, f)
    
    file_size = os.path.getsize(paper_ids_file) / 1024 / 1024
    print(f"✓ Saved {paper_ids_file} ({file_size:.1f} MB)")
    print(f"  Paper IDs: {len(paper_id_set):,}")
    
    print("TRAINING CACHE STATISTICS")

    degrees = [len(edges) for edges in edge_cache.values()]
    print(f"Papers: {len(papers):,}")
    print(f"Citations: {len(citations):,}")
    print(f"Avg degree: {sum(degrees)/len(degrees):.1f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    
    # Author statistics
    total_authors = sum(len(p.get('authors', [])) for p in papers)
    unique_authors = len(set(
        a['authorId'] 
        for p in papers 
        for a in p.get('authors', []) 
        if a.get('authorId')
    ))
    
    print(f"\nAuthor Statistics:")
    print(f"  Total author entries: {total_authors:,}")
    print(f"  Unique authors: {unique_authors:,}")
    print(f"  Avg authors per paper: {total_authors/len(papers):.1f}")
    
    field_counter = defaultdict(int)
    for p in papers:
        fields = p.get('fieldsOfStudy') or p.get('fields') or []
        if isinstance(fields, list):
            for field in fields[:1]:
                if field:
                    field_counter[str(field)] += 1
    
    print(f"\nTop 10 Fields:")
    for i, (field, count) in enumerate(sorted(field_counter.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i:2d}. {field[:30]:30s} {count:6,d} papers")
    
    print("✓ TRAINING CACHE READY")
    print(f"Location: {cache_dir}/")
    print(f"Files:")
    print(f"  - training_papers_1M.pkl")
    print(f"  - edge_cache_1M.pkl")
    print(f"  - paper_id_set_1M.pkl")


if __name__ == '__main__':
    main()
