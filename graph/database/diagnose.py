"""
Diagnose why communities aren't being assigned to papers.
"""

import asyncio
from graph.database.store import EnhancedStore
from graph.database.comm_det import CommunityDetector


async def diagnose():
    print("=" * 80)
    print("COMMUNITY ASSIGNMENT DIAGNOSIS")
    print("=" * 80)
    
    store = EnhancedStore()
    detector = CommunityDetector(store)
    
    # Load cache
    if not detector.load_cache():
        print("✗ No cache found!")
        return
    
    print(f"\n1. Cache loaded: {len(detector.communities)} nodes in cache")
    
    # Get sample cached node IDs
    sample_cached_ids = list(detector.communities.keys())[:5]
    print(f"\n2. Sample cached node IDs:")
    for i, node_id in enumerate(sample_cached_ids, 1):
        comm_id = detector.communities[node_id]
        print(f"   {i}. {node_id}")
        print(f"      → Community: {comm_id}")
    
    # Get actual paper IDs from database
    print(f"\n3. Getting actual paper IDs from database...")
    query = """
        MATCH (p:Paper)
        RETURN elementId(p) as paper_id, 
               COALESCE(p.title, p.id, '') as title
        LIMIT 5
    """
    
    actual_papers = await store._run_query_method(query, [])
    
    print(f"\n4. Sample actual paper IDs:")
    for i, paper in enumerate(actual_papers, 1):
        paper_id = paper.get('paper_id')
        title = paper.get('title', 'Unknown')[:50]
        
        # Check if this ID is in cache
        comm_id = detector.get_community(paper_id)
        
        print(f"   {i}. {paper_id}")
        print(f"      Title: {title}")
        print(f"      Community: {comm_id if comm_id else '❌ NOT IN CACHE'}")
    
    # Check format match
    print(f"\n5. ID Format Analysis:")
    
    if sample_cached_ids and actual_papers:
        cached_format = sample_cached_ids[0]
        actual_format = actual_papers[0].get('paper_id')
        
        print(f"   Cached ID example:  {cached_format}")
        print(f"   Actual ID example:  {actual_format}")
        
        if cached_format == actual_format:
            print(f"   ✓ Formats match!")
        else:
            print(f"   ✗ FORMAT MISMATCH!")
            print(f"      This is why communities aren't found.")
    
    # Check coverage
    print(f"\n6. Coverage Analysis:")
    
    query = """
        MATCH (p:Paper)
        RETURN count(p) as total_papers
    """
    result = await store._run_query_method(query, [])
    total_papers = result[0].get('total_papers', 0) if result else 0
    
    cached_papers = len(detector.communities)
    coverage = (cached_papers / total_papers * 100) if total_papers > 0 else 0
    
    print(f"   Total papers in DB: {total_papers}")
    print(f"   Papers in cache: {cached_papers}")
    print(f"   Coverage: {coverage:.1f}%")
    
    if coverage < 50:
        print(f"   ⚠ Low coverage! Only {coverage:.1f}% of papers have communities.")
    
    # Test with well-connected paper
    print(f"\n7. Testing with well-connected paper:")
    
    paper = await store.get_well_connected_paper()
    if paper:
        paper_id = paper.get('paper_id')
        title = paper.get('title', 'Unknown')[:50]
        comm_id = detector.get_community(paper_id)
        
        print(f"   Paper: {title}")
        print(f"   ID: {paper_id}")
        print(f"   Community: {comm_id if comm_id else '❌ NOT IN CACHE'}")
        
        if not comm_id:
            print(f"\n   ✗ PROBLEM CONFIRMED: Well-connected papers not in cache!")
    
    # Solution recommendation
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    matches = sum(1 for p in actual_papers if detector.get_community(p.get('paper_id')))
    match_rate = (matches / len(actual_papers) * 100) if actual_papers else 0
    
    print(f"\nMatch rate: {matches}/{len(actual_papers)} ({match_rate:.0f}%)")
    
    if match_rate == 0:
        print(f"\n❌ ISSUE: No papers in cache match actual paper IDs")
        print(f"\n💡 SOLUTION:")
        print(f"   The cache was built with wrong node IDs.")
        print(f"   Delete communities.pkl and rebuild:")
        print(f"   ")
        print(f"   rm communities.pkl")
        print(f"   python -m RL.community_detection")
    elif match_rate < 50:
        print(f"\n⚠ ISSUE: Low match rate ({match_rate:.0f}%)")
        print(f"\n💡 SOLUTION:")
        print(f"   Cache has limited coverage. Rebuild with more nodes:")
        print(f"   ")
        print(f"   rm communities.pkl")
        print(f"   python -m RL.community_detection")
    else:
        print(f"\n✓ Cache looks good!")
    
    await store.pool.close()


if __name__ == "__main__":
    asyncio.run(diagnose())