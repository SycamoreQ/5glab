

async def diagnose_graph_sparsity(store):
    """Check citation graph connectivity."""
    
    # 1. Check if CITES relationships exist
    query1 = """
    MATCH ()-[r:CITES]->()
    RETURN count(r) as total_citations
    """
    result1 = await store.run_query_method(query1)
    total_citations = result1[0]['total_citations'] if result1 else 0
    
    # 2. Check papers with 0 citations
    query2 = """
    MATCH (p:Paper)
    WHERE NOT exists((p)-[:CITES]->())
      AND NOT exists(()-[:CITES]->(p))
    RETURN count(p) as isolated_papers
    """
    result2 = await store.run_query_method(query2)
    isolated = result2[0]['isolated_papers'] if result2 else 0
    
    # 3. Check average degree
    query3 = """
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[:CITES]->(ref)
    OPTIONAL MATCH (citing)-[:CITES]->(p)
    WITH p, count(DISTINCT ref) as out_degree, count(DISTINCT citing) as in_degree
    RETURN 
        avg(out_degree + in_degree) as avg_degree,
        max(out_degree + in_degree) as max_degree,
        min(out_degree + in_degree) as min_degree
    """
    result3 = await store.run_query_method(query3)
    
    # 4. Sample a paper and check neighbors
    query4 = """
    MATCH (p:Paper)
    WHERE p.citationCount > 20
    WITH p LIMIT 1
    OPTIONAL MATCH (p)-[:CITES]->(ref)
    OPTIONAL MATCH (citing)-[:CITES]->(p)
    OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
    RETURN 
        p.paperId as paper_id,
        p.title as title,
        count(DISTINCT ref) as references,
        count(DISTINCT citing) as citations,
        count(DISTINCT a) as authors
    """
    result4 = await store.run_query_method(query4)
    
    print("\n" + "="*70)
    print("GRAPH CONNECTIVITY DIAGNOSIS")
    print("="*70)
    print(f"Total CITES edges: {total_citations:,}")
    print(f"Isolated papers (no edges): {isolated:,}")
    
    if result3:
        print(f"\nDegree statistics:")
        print(f"  Average: {result3[0]['avg_degree']:.2f}")
        print(f"  Max: {result3[0]['max_degree']}")
        print(f"  Min: {result3[0]['min_degree']}")
    
    if result4:
        print(f"\nSample paper:")
        print(f"  Title: {result4[0]['title'][:60]}")
        print(f"  References: {result4[0]['references']}")
        print(f"  Citations: {result4[0]['citations']}")
        print(f"  Authors: {result4[0]['authors']}")
    
    print("="*70 + "\n")
    
    # Verdict
    if total_citations == 0:
        print("⚠️  CRITICAL: No CITES relationships found!")
        print("   Your citation graph is EMPTY. You need to:")
        print("   1. Re-run your data loader with citation edges")
        print("   2. Check if your source data includes references/citations")
    elif result3 and result3[0]['avg_degree'] < 2:
        print("⚠️  WARNING: Very sparse graph (avg degree < 2)")
        print("   Most papers have <1 neighbor on average")
    elif result3 and result3[0]['avg_degree'] < 5:
        print("⚠️  Graph is sparse but workable (avg degree 2-5)")
    else:
        print("✓ Graph connectivity looks good")

# Add to train() function:
async def train():
    store = EnhancedStore()
    
    # Diagnose first!
    await diagnose_graph_sparsity(store)
    
    # ... rest of training code ...
