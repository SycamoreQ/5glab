import asyncio
from graph.database.store import EnhancedStore

async def check_years():
    store = EnhancedStore()
    
    query = """
    MATCH (p:Paper)
    WHERE p.year IS NOT NULL
    RETURN count(p) as papers_with_year,
           min(p.year) as min_year,
           max(p.year) as max_year,
           avg(p.year) as avg_year
    """
    result = await store._run_query_method(query, [])
    print("Papers with year data:", result)
    
    query2 = """
    MATCH (p:Paper)
    WHERE p.year IS NULL
    RETURN count(p) as papers_without_year
    """
    result2 = await store._run_query_method(query2, [])
    print("Papers without year:", result2)
    
    await store.pool.close()

if __name__ == "__main__":
    asyncio.run(check_years())
