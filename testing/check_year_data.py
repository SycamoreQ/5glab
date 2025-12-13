import asyncio
from graph.database.store import EnhancedStore

async def diagnose():
    store = EnhancedStore()
    
    # Check CITES relationships
    query1 = """
    MATCH ()-[r:CITES]->()
    RETURN count(r) as cites_edges
    """
    result1 = await store._run_query_method(query1)
    cites = result1[0]['cites_edges'] if result1 else 0
    
    # Check WROTE relationships
    query2 = """
    MATCH ()-[r:WROTE]->()
    RETURN count(r) as wrote_edges
    """
    result2 = await store._run_query_method(query2)
    wrote = result2[0]['wrote_edges'] if result2 else 0
    
    # Check isolated papers
    query3 = """
    MATCH (p:Paper)
    WHERE NOT exists((p)-[:CITES]->()) 
      AND NOT exists(()-[:CITES]->(p))
    RETURN count(p) as isolated
    """
    result3 = await store._run_query_method(query3)
    isolated = result3[0]['isolated'] if result3 else 0
    
    # Check paper count
    query4 = """
    MATCH (p:Paper)
    RETURN count(p) as total_papers
    """
    result4 = await store._run_query_method(query4)
    total_papers = result4[0]['total_papers'] if result4 else 0
    
    print("\n" + "="*70)
    print("GRAPH STRUCTURE ANALYSIS")
    print("="*70)
    print(f"Total Papers: {total_papers:,}")
    print(f"CITES edges: {cites:,}")
    print(f"WROTE edges: {wrote:,}")
    print(f"Isolated papers (no citations): {isolated:,} ({100*isolated/total_papers:.1f}%)")
    print(f"\nAverage degree: {2*cites/total_papers:.2f}")
    print("="*70 + "\n")
    
    if cites < total_papers * 5:
        print("⚠️  CRITICAL: Very few citation edges!")
        print("   Expected: ~200-500M CITES edges for 44M papers")
        print(f"   Actual: {cites:,} CITES edges")
        print("\n   Your citation loading failed or is incomplete.")

if __name__ == '__main__':
    asyncio.run(diagnose())
