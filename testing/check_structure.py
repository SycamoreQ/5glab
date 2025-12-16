"""
Test connectivity of training graph.
"""
import pickle
import random

def test_connectivity():
    with open('training_cache/edge_cache_1M.pkl', 'rb') as f:
        edge_cache = pickle.load(f)
    
    with open('training_cache/training_papers_1M.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    print(f"Testing {len(papers):,} papers...")
    
    # Sample 1000 papers
    sample = random.sample(papers, min(1000, len(papers)))
    
    dead_ends = 0
    degrees = []
    
    for paper in sample:
        pid = paper.get('paperId') or paper.get('paper_id')
        neighbors = edge_cache.get(pid, [])
        degree = len(neighbors)
        degrees.append(degree)
        
        if degree == 0:
            dead_ends += 1
    
    print(f"\nResults:")
    print(f"  Dead ends: {dead_ends}/{len(sample)} ({100*dead_ends/len(sample):.1f}%)")
    print(f"  Avg degree: {sum(degrees)/len(degrees):.2f}")
    print(f"  Papers with 0 neighbors: {sum(1 for d in degrees if d == 0)}")
    print(f"  Papers with 1-2 neighbors: {sum(1 for d in degrees if 1 <= d <= 2)}")
    print(f"  Papers with 3-5 neighbors: {sum(1 for d in degrees if 3 <= d <= 5)}")
    print(f"  Papers with >5 neighbors: {sum(1 for d in degrees if d > 5)}")

if __name__ == '__main__':
    test_connectivity()
