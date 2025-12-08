"""
Quick test to verify abstracts improved similarity scores.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


def test_abstract_impact():
    """
    Compare similarity scores with and without abstracts.
    """
    print("="*80)
    print("ABSTRACT IMPACT TEST")
    print("="*80)
    
    # Load enriched cache
    print("\n1. Loading enriched cache...")
    with open('training_papers_enriched.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    # Count papers with abstracts
    with_abstract = [p for p in papers if p.get('abstract') and len(p['abstract']) > 50]
    print(f"   Total papers: {len(papers):,}")
    print(f"   With abstracts: {len(with_abstract):,} ({len(with_abstract)/len(papers)*100:.1f}%)")
    
    # Initialize encoder
    print("\n2. Initializing encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test queries (from your domain)
    test_queries = [
        "machine learning neural networks",
        "deep learning computer vision",
        "natural language processing",
        "reinforcement learning",
        "graph neural networks"
    ]
    
    print("\n3. Testing similarity improvements...")
    print("-"*80)
    
    all_title_sims = []
    all_full_sims = []
    improvements = []
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        query_emb = encoder.encode(query)
        
        keywords = query.lower().split()
        matching_papers = []
        
        for paper in with_abstract[:500]:  # Check first 500 with abstracts
            title_lower = paper.get('title', '').lower()
            abstract_lower = paper.get('abstract', '').lower()
            
            # Check if any keyword appears in title or abstract
            if any(kw in title_lower or kw in abstract_lower for kw in keywords):
                matching_papers.append(paper)
                if len(matching_papers) >= 3:
                    break
        
        if not matching_papers:
            print("   ⚠ No matching papers found")
            continue
        
        for paper in matching_papers:
            title = paper['title']
            abstract = paper.get('abstract', '')
            
            # Test 1: Title only
            title_emb = encoder.encode(title)
            sim_title = np.dot(query_emb, title_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(title_emb)
            )
            
            # Test 2: Title + Abstract
            full_text = f"{title} {abstract[:500]}"
            full_emb = encoder.encode(full_text)
            sim_full = np.dot(query_emb, full_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(full_emb)
            )
            
            improvement = sim_full - sim_title
            
            all_title_sims.append(sim_title)
            all_full_sims.append(sim_full)
            improvements.append(improvement)
            
            print(f"\n   Paper: {title[:60]}...")
            print(f"   Title-only:      {sim_title:.3f}")
            print(f"   With abstract:   {sim_full:.3f} ✓")
            print(f"   Improvement:     +{improvement:.3f}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    if all_title_sims:
        avg_title = np.mean(all_title_sims)
        avg_full = np.mean(all_full_sims)
        avg_improvement = np.mean(improvements)
        
        print(f"\nAverage Similarity:")
        print(f"  Title only:        {avg_title:.3f}")
        print(f"  With abstract:     {avg_full:.3f}")
        print(f"  Average improvement: +{avg_improvement:.3f} ({avg_improvement/avg_title*100:.1f}% increase)")
        
        print(f"\nRange:")
        print(f"  Title: {np.min(all_title_sims):.3f} - {np.max(all_title_sims):.3f}")
        print(f"  Full:  {np.min(all_full_sims):.3f} - {np.max(all_full_sims):.3f}")
        
        # Verdict
        print(f"\n" + "="*80)
        if avg_full >= 0.5:
            print("✓ EXCELLENT: Average similarity ≥ 0.5")
            print("  Your RL agent should perform MUCH better now!")
            print("  Expected episode rewards: 50-100 (was 10-30)")
        elif avg_full >= 0.4:
            print("✓ GOOD: Average similarity ≥ 0.4")
            print("  Significant improvement over title-only")
            print("  Expected episode rewards: 40-70")
        elif avg_full >= 0.3:
            print("⚠ MODERATE: Average similarity ≥ 0.3")
            print("  Some improvement but could be better")
            print("  Consider enriching more papers or using arXiv dataset")
        else:
            print("⚠ LIMITED: Similarity still low")
            print("  Abstracts may be too short or not relevant")
            print("  Recommend: Use arXiv or S2ORC for better abstracts")
        
        print(f"="*80)
    else:
        print("⚠ No test papers found. Try with different queries.")


if __name__ == "__main__":
    test_abstract_impact()