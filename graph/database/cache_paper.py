import pickle
import os
from collections import Counter


def export_training_cache():
    print("Loading pruned graph...")
    with open('pruned_graph_enhanced_1M.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    
    papers = graph_data['papers']
    citations = graph_data['citations']
    
    print(f"Total papers: {len(papers):,}")
    print(f"Citations: {len(citations):,}")
    
    print("\nLoading embeddings to filter valid papers...")
    embeddings_file = 'training_cache/embeddings_1M.pkl'
    
    if not os.path.exists(embeddings_file):
        print(f"ERROR: {embeddings_file} not found!")
        print("Run embedding generation first, then rebuild cache.")
        return
    
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    embedded_ids = {str(k) for k in embeddings.keys()}
    print(f"Loaded {len(embedded_ids):,} embeddings")
    

    all_papers = papers
    filtered_papers = [
        p for p in all_papers 
        if str(p.get('paperId')) in embedded_ids
    ]
    print(f"Filtered papers: {len(all_papers):,} -> {len(filtered_papers):,}")
    
    edge_cache = {}
    valid_edges = 0
    skipped_edges = 0
    
    print("\nBuilding edge cache...")
    for cite in citations:
        source = str(cite['source'])
        target = str(cite['target'])

        if source not in embedded_ids or target not in embedded_ids:
            skipped_edges += 1
            continue
        
        if source not in edge_cache:
            edge_cache[source] = []
        if target not in edge_cache:
            edge_cache[target] = []
        
        edge_cache[source].append(('cites', target))
        edge_cache[target].append(('cited_by', source))
        valid_edges += 2  
    
    paper_id_set = {str(p['paperId']) for p in filtered_papers if p.get('paperId')}
    
    papers_with_edges = {pid for pid in edge_cache if edge_cache[pid]}
    final_filtered_papers = [
        p for p in filtered_papers 
        if str(p.get('paperId')) in papers_with_edges
    ]
    
    print(f"\nEdge cache statistics:")
    print(f"  Total citations processed: {len(citations):,}")
    print(f"  Valid edges (both endpoints embedded): {valid_edges:,}")
    print(f"  Skipped edges (missing embeddings): {skipped_edges:,}")
    print(f"  Papers with edges: {len(papers_with_edges):,}")
    print(f"  Final filtered papers: {len(final_filtered_papers):,}")

    degrees = Counter(len(edges) for edges in edge_cache.values())
    print(f"\nDegree distribution (top 15):")
    for d in sorted(degrees.keys())[:15]:
        print(f"  Degree {d}: {degrees[d]:,} papers")

    isolated = sum(1 for d in degrees.values() if d == 0)
    print(f"\n  Papers with degree 0: {isolated:,}")
    print(f"  Papers with degree 1: {degrees.get(1, 0):,}")
    print(f"  Papers with degree 2+: {len(edge_cache) - isolated - degrees.get(1, 0):,}")
    
    if edge_cache:
        print("\nTesting connectivity...")
        visited = set()
        
        def bfs(start, limit=100000):
            queue = [start]
            count = 0
            while queue and count < limit:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                count += 1
                if node in edge_cache:
                    for _, neighbor in edge_cache[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            return count
        
        # Find a node with edges to start BFS
        degrees_list = [(len(edges), pid) for pid, edges in edge_cache.items() if edges]
        if not degrees_list:
            print("  No papers with edges!")
        else:
            degrees_list.sort(reverse=True)
            start_node = degrees_list[0][1]  
            print(f"  Starting BFS from paper with degree {degrees_list[0][0]}")
        component_size = bfs(start_node, limit=100000)
        print(f"  Largest component: {component_size:,} papers")
        
        coverage = (component_size / len(papers_with_edges) * 100) if papers_with_edges else 0
        print(f"  Coverage: {coverage:.1f}% of papers with edges")
    
    # Final validation
    print(f"\n VALIDATION:")
    papers_missing_embeddings = sum(
        1 for p in final_filtered_papers 
        if str(p.get('paperId')) not in embedded_ids
    )
    print(f"papers missing embeddings: {papers_missing_embeddings}")
    
    edges_with_missing_embeddings = sum(
        1 for edges in edge_cache.values()
        for _, target in edges
        if target not in embedded_ids
    )
    print(f"  Edges with missing embeddings: {edges_with_missing_embeddings}")
    
    if papers_missing_embeddings > 0 or edges_with_missing_embeddings > 0:
        print("\nWARNING: Some papers or edges are missing embeddings!")
    else:
        print("\nAll papers and edges have valid embeddings!")
    
    print(f"\nExporting training cache...")
    os.makedirs('training_cache', exist_ok=True)
    
    with open('training_cache/training_papers_1M.pkl', 'wb') as f:
        pickle.dump(final_filtered_papers, f)
    print(f"  ✓ Saved training_papers_1M.pkl")
    
    with open('training_cache/edge_cache_1M.pkl', 'wb') as f:
        pickle.dump(edge_cache, f)
    print(f"  ✓ Saved edge_cache_1M.pkl")
    
    with open('training_cache/paper_id_set_1M.pkl', 'wb') as f:
        pickle.dump(paper_id_set, f)
    print(f"  ✓ Saved paper_id_set_1M.pkl")
    
    print("TRAINING CACHE EXPORTED!")
    print("="*70)
    print(f"  Papers: {len(final_filtered_papers):,}")
    print(f"  Edges: {valid_edges:,}")
    print(f"  Paper IDs: {len(paper_id_set):,}")
    print(f"  Avg degree: {valid_edges / len(papers_with_edges):.1f}" if papers_with_edges else "  Avg degree: 0")

if __name__ == '__main__':
    export_training_cache()
