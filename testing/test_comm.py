import pickle
from collections import Counter
from graph.database.comm_det import CommunityDetector

def analyze_community_distribution():
    """Analyze current community skew."""
    
    detector = CommunityDetector(cache_file='community_cache_1M.pkl')
    if not detector.load_cache():
        print("❌ Failed to load cache")
        return
    
    # Get distribution
    paper_comms = list(detector.paper_communities.values())
    comm_counts = Counter(paper_comms)
    
    print(f"\n{'='*70}")
    print(f"COMMUNITY DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    total = len(paper_comms)
    print(f"Total papers: {total:,}\n")
    
    # Top 20 communities
    print("Top 20 Largest Communities:")
    print(f"{'Community':<25} {'Papers':>10} {'%':>6}")
    print("-" * 70)
    
    for comm, count in comm_counts.most_common(20):
        pct = 100 * count / total
        print(f"{str(comm):<25} {count:>10,} {pct:>5.1f}%")
    
    # Statistics
    print(f"\n{'='*70}")
    misc_count = sum(count for comm, count in comm_counts.items() 
                     if 'MISC' in str(comm))
    misc_pct = 100 * misc_count / total
    
    print(f"MISC communities: {misc_pct:.1f}% of all papers")
    print(f"Unique communities: {len(comm_counts):,}")
    print(f"Median community size: {sorted(comm_counts.values())[len(comm_counts)//2]}")
    print(f"Mean community size: {total / len(comm_counts):.1f}")
    print(f"{'='*70}\n")
    
    # Recommendations
    if misc_pct > 50:
        print("⚠️  WARNING: Over 50% of papers in MISC communities!")
        print("   This will cause severe ping-pong behavior.\n")
        print("💡 Recommendations:")
        print("   1. Lower min_community_size from 50 to 20")
        print("   2. Use Louvain instead of Label Propagation")
        print("   3. Increase resolution parameter")
        print("   4. Filter by citation count > 5")

if __name__ == '__main__':
    analyze_community_distribution()
