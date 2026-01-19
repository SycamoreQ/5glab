import requests
import time
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_abstracts_batch(paper_ids, api_key):
    """Fetch abstracts for a batch of papers."""
    results = []
    
    for paper_id in paper_ids:
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusID:{paper_id}"
            headers = {'x-api-key': api_key} if api_key else {}
            params = {'fields': 'abstract'}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                abstract = data.get('abstract')
                if abstract:
                    results.append({'paperId': paper_id, 'abstract': abstract})
            
            time.sleep(0.05)  # 20 requests/second with API key
            
        except Exception as e:
            logger.error(f"Error fetching {paper_id}: {e}")
    
    return results

def update_abstracts_from_api(neo4j_uri, neo4j_user, neo4j_password, api_key, limit=50000):
    """Fetch and update abstracts using Semantic Scholar API."""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Get papers without abstracts (prioritize recent papers)
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.abstract IS NULL AND p.year >= 2020
            RETURN p.paperId as paperId
            ORDER BY p.year DESC, p.citationCount DESC
            LIMIT $limit
        """, limit=limit)
        
        paper_ids = [record['paperId'] for record in result]
    
    logger.info(f"Fetching abstracts for {len(paper_ids)} papers (2020+)...")
    
    # Process in batches
    batch_size = 100
    total_updated = 0
    
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i+batch_size]
        results = fetch_abstracts_batch(batch, api_key)
        
        # Update database
        if results:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (p:Paper {paperId: item.paperId})
                    SET p.abstract = item.abstract,
                        p.lastUpdated = datetime()
                """, batch=results)
            
            total_updated += len(results)
            logger.info(f"Progress: {i+batch_size}/{len(paper_ids)}, Updated: {total_updated}")
    
    driver.close()
    logger.info(f"Total abstracts fetched: {total_updated}")
    return total_updated

# Usage
if __name__ == "__main__":
    updated = update_abstracts_from_api(
        neo4j_uri='bolt://localhost:7687',
        neo4j_user='neo4j',
        neo4j_password='diam0ndman@3',
        api_key='7t7DOixHB88DPE70ztPCN4cyYSp8K9aj8kYGgIFx',
        limit=50000  # Start with 50k papers
    )
