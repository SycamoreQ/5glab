"""
Diagnose why abstracts are making similarity worse.
"""

import pickle
import numpy as np


def diagnose_abstracts():
    """
    Check abstract quality in enriched cache.
    """
    print("="*80)
    print("ABSTRACT QUALITY DIAGNOSIS")
    print("="*80)
    
    # Load enriched cache
    with open('training_papers_enriched.pkl', 'rb') as f:
        papers = pickle.load(f)
    
    with_abstract = [p for p in papers if p.get('abstract') and len(p['abstract']) > 50]
    
    print(f"\n1. Basic Statistics:")
    print(f"   Total papers: {len(papers):,}")
    print(f"   With abstracts: {len(with_abstract):,} ({len(with_abstract)/len(papers)*100:.1f}%)")
    
    # Check abstract lengths
    abstract_lengths = [len(p['abstract']) for p in with_abstract]
    avg_length = np.mean(abstract_lengths)
    min_length = np.min(abstract_lengths)
    max_length = np.max(abstract_lengths)
    median_length = np.median(abstract_lengths)
    
    print(f"\n2. Abstract Length Analysis:")
    print(f"   Average: {avg_length:.0f} characters")
    print(f"   Median:  {median_length:.0f} characters")
    print(f"   Min:     {min_length:.0f} characters")
    print(f"   Max:     {max_length:.0f} characters")
    
    # Length distribution
    very_short = sum(1 for l in abstract_lengths if l < 200)
    short = sum(1 for l in abstract_lengths if 200 <= l < 500)
    medium = sum(1 for l in abstract_lengths if 500 <= l < 1000)
    long = sum(1 for l in abstract_lengths if l >= 1000)
    
    print(f"\n3. Length Distribution:")
    print(f"   <200 chars (very short):  {very_short:>5} ({very_short/len(with_abstract)*100:.1f}%)")
    print(f"   200-500 chars (short):    {short:>5} ({short/len(with_abstract)*100:.1f}%)")
    print(f"   500-1000 chars (medium):  {medium:>5} ({medium/len(with_abstract)*100:.1f}%)")
    print(f"   >1000 chars (long):       {long:>5} ({long/len(with_abstract)*100:.1f}%)")
    
    # Show samples
    print(f"\n4. Sample Abstracts (checking quality):")
    print("-"*80)
    
    # Show a few abstracts
    for i, paper in enumerate(with_abstract[:5], 1):
        title = paper['title']
        abstract = paper['abstract']
        
        print(f"\nPaper {i}:")
        print(f"Title: {title[:70]}...")
        print(f"Abstract length: {len(abstract)} chars")
        print(f"Abstract preview:")
        print(f"  {abstract[:200]}...")
        
        # Check if abstract seems relevant
        title_words = set(title.lower().split())
        abstract_words = set(abstract.lower().split())
        overlap = title_words.intersection(abstract_words)
        overlap_ratio = len(overlap) / len(title_words) if title_words else 0
        
        print(f"Word overlap with title: {len(overlap)}/{len(title_words)} ({overlap_ratio*100:.1f}%)")
        
        # Check for common issues
        issues = []
        if len(abstract) < 200:
            issues.append("Too short")
        if abstract.count('.') < 2:
            issues.append("Single sentence")
        if 'http' in abstract or 'www.' in abstract:
            issues.append("Contains URLs")
        if abstract.lower().startswith('abstract'):
            issues.append("Starts with 'Abstract'")
        
        if issues:
            print(f"⚠ Issues: {', '.join(issues)}")
        else:
            print(f"✓ Looks OK")
    
    # Analysis
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if avg_length < 300:
        print("\n❌ PROBLEM: Abstracts are too short (avg <300 chars)")
        print("   OpenAlex abstracts are often truncated or incomplete.")
        print("\n   SOLUTION: Use arXiv dataset instead")
        print("   - arXiv has full, high-quality abstracts")
        print("   - 2M papers, 100% coverage")
        print("   - Download: kaggle datasets download -d Cornell-University/arxiv")
    
    elif very_short / len(with_abstract) > 0.3:
        print("\n⚠ PROBLEM: Too many very short abstracts (>30%)")
        print("   Many abstracts are incomplete or just first sentence.")
        print("\n   SOLUTION: Try S2ORC or arXiv for better quality")
    
    else:
        print("\n⚠ PROBLEM: Abstracts seem OK but similarity still decreased")
        print("   This might be due to:")
        print("   1. Inverted index reconstruction issues")
        print("   2. Abstracts from wrong papers (matching errors)")
        print("   3. Generic abstracts that dilute specific titles")
        print("\n   SOLUTION: Use a dataset with verified paper-abstract pairs")
        print("   - Recommended: arXiv (highest quality)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("\n🎯 Download arXiv dataset for guaranteed improvement:")
    print("\n   Step 1: Download")
    print("   $ kaggle datasets download -d Cornell-University/arxiv")
    print("   $ unzip arxiv.zip")
    print("\n   Step 2: Import arXiv papers")
    print("   $ python import_arxiv.py")
    print("\n   Step 3: Rebuild cache with arXiv papers")
    print("   $ python rebuild_training_cache.py")
    print("\n   Expected improvement: 0.25 → 0.55+ similarity")
    print("   Time: 2-3 hours total")
    print("\n" + "="*80)


if __name__ == "__main__":
    diagnose_abstracts()