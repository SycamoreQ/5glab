"""
Script to check what properties actually exist on your Paper nodes
and find a valid identifier to use for training.
"""
import asyncio
from graph.database.store import EnhancedStore

async def check_database():
    store = EnhancedStore()
    
    print("=" * 80)
    print("CHECKING NEO4J DATABASE SCHEMA")
    print("=" * 80)
    
    # Method 1: Get ALL properties of a sample paper
    print("\n1. Checking what properties exist on Paper nodes...")
    query = """
        MATCH (p:Paper)
        RETURN p
        LIMIT 1
    """
    result = await store._run_query_method(query, [])
    
    if result and result[0]:
        paper_node = result[0].get('p', {})
        print(f"\n✓ Sample Paper Node properties:")
        for key, value in paper_node.items():
            value_preview = str(value)[:80] if value else "None"
            print(f"    {key}: {value_preview}")
    else:
        print("✗ No papers found!")
        return
    
    # Method 2: Check what can be used as unique identifier
    print("\n" + "=" * 80)
    print("2. Checking potential unique identifiers...")
    
    # Check if 'id' property exists
    query = """
        MATCH (p:Paper)
        WHERE p.id IS NOT NULL
        RETURN p.id, p.title
        LIMIT 3
    """
    result = await store._run_query_method(query, [])
    
    if result:
        print(f"\n✓ Papers have 'id' property:")
        for paper in result[:3]:
            print(f"    id: {paper.get('p.id')}")
            print(f"    title: {paper.get('p.title', 'No title')[:60]}...")
    else:
        print("\n✗ No papers with 'id' property")
    
    # Check if we can use Neo4j internal ID
    print("\n" + "=" * 80)
    print("3. Using Neo4j internal ID (elementId)...")
    query = """
        MATCH (p:Paper)
        RETURN elementId(p) as node_id, p.title, p.doi
        LIMIT 5
    """
    result = await store._run_query_method(query, [])
    
    if result:
        print(f"\n✓ Sample papers with elementId:")
        for i, paper in enumerate(result, 1):
            node_id = paper.get('node_id')
            title = paper.get('p.title', 'No title')
            doi = paper.get('p.doi', 'No DOI')
            print(f"\n  Paper {i}:")
            print(f"    elementId: {node_id}")
            print(f"    Title: {title[:60]}...")
            print(f"    DOI: {doi}")
            
            if i == 1:
                print(f"\n  → RECOMMENDED: Use this as starting node:")
                print(f"    START_NODE_ID = \"{node_id}\"")
    
    # Method 4: Find well-connected papers using elementId
    print("\n" + "=" * 80)
    print("4. Finding well-connected papers (using elementId)...")
    query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        WITH p, elementId(p) as node_id, count(DISTINCT ref) as ref_count, count(DISTINCT citing) as cite_count
        WHERE ref_count > 0 AND cite_count > 0
        RETURN node_id, p.title, p.doi, ref_count, cite_count
        ORDER BY (ref_count + cite_count) DESC
        LIMIT 5
    """
    result = await store._run_query_method(query, [])
    
    if result:
        print(f"\n✓ Found {len(result)} well-connected papers:")
        for i, paper in enumerate(result, 1):
            node_id = paper.get('node_id')
            title = paper.get('p.title', 'No title')
            doi = paper.get('p.doi', 'No DOI')
            ref_count = paper.get('ref_count', 0)
            cite_count = paper.get('cite_count', 0)
            
            print(f"\n  Paper {i}:")
            print(f"    elementId: {node_id}")
            print(f"    Title: {title[:70]}...")
            print(f"    DOI: {doi}")
            print(f"    References: {ref_count}, Citations: {cite_count}")
    
    # Method 5: Check if DOI search works
    print("\n" + "=" * 80)
    print("5. Testing DOI lookup...")
    test_doi = "10.1016/j.ijcce.2025.10.010"
    query = """
        MATCH (p:Paper)
        WHERE p.doi = $1
        RETURN elementId(p) as node_id, p.title, p.doi
        LIMIT 1
    """
    result = await store._run_query_method(query, [test_doi])
    
    if result:
        paper = result[0]
        print(f"\n✓ Found paper with DOI: {test_doi}")
        print(f"    elementId: {paper.get('node_id')}")
        print(f"    Title: {paper.get('p.title', 'No title')}")
    else:
        print(f"\n✗ Paper with DOI '{test_doi}' not found")
        
        # Try to find ANY paper with a DOI
        query = """
            MATCH (p:Paper)
            WHERE p.doi IS NOT NULL
            RETURN elementId(p) as node_id, p.title, p.doi
            LIMIT 3
        """
        result = await store._run_query_method(query, [])
        if result:
            print(f"\n  But here are some papers with DOIs:")
            for paper in result:
                print(f"    DOI: {paper.get('p.doi')}")
                print(f"    elementId: {paper.get('node_id')}")
                print(f"    Title: {paper.get('p.title', 'No title')[:60]}...")
                print()
    
    # Method 6: Check title-based lookup
    print("=" * 80)
    print("6. Testing title lookup...")
    query = """
        MATCH (p:Paper)
        WHERE p.title CONTAINS 'Adam'
        RETURN elementId(p) as node_id, p.title, p.doi
        LIMIT 3
    """
    result = await store._run_query_method(query, [])
    
    if result:
        print(f"\n✓ Found papers with 'Adam' in title:")
        for paper in result:
            print(f"    elementId: {paper.get('node_id')}")
            print(f"    Title: {paper.get('p.title')}")
            print()
    
    print("=" * 80)
    print("\nCONCLUSION:")
    print("Your database does NOT have 'paper_id' as a property.")
    print("You need to either:")
    print("  1. Use Neo4j's elementId(p) as the node identifier")
    print("  2. Add a 'paper_id' property to your nodes")
    print("  3. Use 'doi' or 'id' if they exist")
    print("=" * 80)
    
    await store.pool.close()

if __name__ == "__main__":
    asyncio.run(check_database())