import requests
import time
from neo4j import GraphDatabase
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def fetch_abstracts_batch_api(paper_ids: List[str], api_key: str, max_retries: int = 5) -> Dict[str, str]:
    """Fetch abstracts with aggressive retry logic."""
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    corpus_ids = [f"CorpusID:{pid}" for pid in paper_ids]
    params = {'fields': 'corpusId,abstract'}
    payload = {'ids': corpus_ids}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, params=params, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                abstracts = {}
                for paper in results:
                    if paper and paper.get('abstract'):
                        corpus_id = paper.get('corpusId')
                        if corpus_id:
                            abstracts[str(corpus_id)] = paper['abstract']
                return abstracts
            
            elif response.status_code == 429:
                wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                logger.warning(f"Rate limited. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            else:
                logger.error(f"API error {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {}
                
        except Exception as e:
            logger.error(f"Request error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    logger.error(f"Failed after {max_retries} retries")
    return {}


def update_abstracts_sequential(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    api_key: str,
    total_limit: int = 100000,
    api_batch_size: int = 500,
    db_batch_size: int = 5000,
    delay_between_requests: float = 0.2  # 200ms = 5 req/sec
):
    """
    Sequential fetching with strict rate limiting (SAFE & RELIABLE).
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    logger.info("Fetching papers without abstracts from Neo4j...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.abstract IS NULL AND p.year >= 2018
            RETURN p.paperId as paperId
            ORDER BY p.year DESC, p.citationCount DESC
            LIMIT $limit
        """, limit=total_limit)
        
        paper_ids = [record['paperId'] for record in result]
    
    logger.info(f"Found {len(paper_ids)} papers without abstracts")
    logger.info(f"Using sequential processing with {delay_between_requests}s delay (~{int(1/delay_between_requests)} req/sec)")
    logger.info(f"Estimated time: {len(paper_ids) * delay_between_requests / api_batch_size / 60:.1f} minutes")
    
    # Split into batches
    batches = [paper_ids[i:i+api_batch_size] for i in range(0, len(paper_ids), api_batch_size)]
    
    total_updated = 0
    total_processed = 0
    update_buffer = []
    failed_batches = 0
    
    # Process batches sequentially
    for i, batch in enumerate(batches):
        # Fetch abstracts
        abstracts = fetch_abstracts_batch_api(batch, api_key)
        
        if not abstracts:
            failed_batches += 1
        
        for paper_id, abstract in abstracts.items():
            update_buffer.append({
                'paperId': paper_id,
                'abstract': abstract
            })
        
        total_processed += len(batch)
        
        # Update database when buffer is full
        if len(update_buffer) >= db_batch_size:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (p:Paper {paperId: item.paperId})
                    SET p.abstract = item.abstract,
                        p.lastUpdated = datetime()
                """, batch=update_buffer)
            
            total_updated += len(update_buffer)
            
            # Calculate ETA
            progress_pct = 100 * total_processed / len(paper_ids)
            success_rate = 100 * total_updated / max(total_processed, 1)
            batches_remaining = len(batches) - i - 1
            eta_minutes = batches_remaining * delay_between_requests / 60
            
            logger.info(
                f"Progress: {total_processed:,}/{len(paper_ids):,} ({progress_pct:.1f}%), "
                f"{total_updated:,} updated ({success_rate:.1f}% success), "
                f"{failed_batches} failed, ETA: {eta_minutes:.1f}m"
            )
            update_buffer.clear()
        
        # Rate limiting delay
        time.sleep(delay_between_requests)
    
    # Final flush
    if update_buffer:
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MATCH (p:Paper {paperId: item.paperId})
                SET p.abstract = item.abstract,
                    p.lastUpdated = datetime()
            """, batch=update_buffer)
        
        total_updated += len(update_buffer)
    
    driver.close()
    
    logger.info("="*60)
    logger.info(f"COMPLETED: {total_updated:,} abstracts updated out of {total_processed:,} papers")
    logger.info(f"Success rate: {100*total_updated/max(total_processed,1):.1f}%")
    logger.info(f"Failed batches: {failed_batches}")
    logger.info("="*60)
    
    return total_updated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch abstracts sequentially with strict rate limiting')
    parser.add_argument('--api-key', required=True, help='Semantic Scholar API key')
    parser.add_argument('--neo4j-password', required=True, help='Neo4j password')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--limit', type=int, default=500000, help='Max papers to process')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between requests in seconds (default: 0.2)')
    
    args = parser.parse_args()
    
    updated = update_abstracts_sequential(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        api_key=args.api_key,
        total_limit=args.limit,
        delay_between_requests=args.delay
    )
