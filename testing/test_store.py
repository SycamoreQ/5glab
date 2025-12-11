# Quick test script: test_store.py
import asyncio
from graph.database.store import EnhancedStore

async def test():
    store = EnhancedStore()
    
    # Get a random paper
    papers = await store._run_query_method(
        "MATCH (p:Paper) WHERE p.paperId IS NOT NULL RETURN p.paperId LIMIT 1",
        []
    )
    
    if papers:
        paper_id = papers[0]['p.paperId']
        print(f"Testing with paper: {paper_id}")
        
        refs = await store.get_references_by_paper(paper_id)
        cites = await store.get_citations_by_paper(paper_id)
        
        print(f"  References: {len(refs)}")
        print(f"  Citations: {len(cites)}")
    else:
        print("No papers found!")
    
    await store.pool.close()

asyncio.run(test())
