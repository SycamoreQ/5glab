import requests
import time
import argparse
import os
import sys
from typing import List, Dict, Optional, Set
from neo4j import GraphDatabase
import logging
from datetime import datetime
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 0.9):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Max requests per second (default 0.9 to stay under 1 RPS limit)
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            sleep_time = self.min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class SemanticScholarAPI:
    """Wrapper for Semantic Scholar API with rate limiting."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None, requests_per_second: float = 0.9, max_retries: int = 3):
        """
        Initialize API client.
        
        Args:
            api_key: Optional S2 API key for higher rate limits
            requests_per_second: Rate limit (default 0.9 for safety margin)
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.rate_limiter = RateLimiter(requests_per_second)
        self.session = requests.Session()
        self.max_retries = max_retries
        
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
            logger.info("API key configured")
        else:
            logger.warning("No API key provided - using unauthenticated access with shared rate limits")
    
    def _make_request(self, url: str, params: Dict, retry_count: int = 0) -> Optional[Dict]:
        """Make rate-limited API request with error handling and retries."""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, params=params, timeout=60)  # Increased timeout
            
            if response.status_code == 200:
                return response.json()
                
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(url, params, retry_count)
                
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return None
                
            elif response.status_code == 504:
                # Gateway timeout - retry with exponential backoff
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count * 5  # 5, 10, 20 seconds
                    logger.warning(f"API timeout (504). Retry {retry_count + 1}/{self.max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                    return self._make_request(url, params, retry_count + 1)
                else:
                    logger.error(f"API timeout after {self.max_retries} retries. Skipping this request.")
                    return None
                    
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count * 5
                logger.warning(f"Request timeout. Retry {retry_count + 1}/{self.max_retries} after {wait_time}s...")
                time.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)
            else:
                logger.error(f"Request timeout after {self.max_retries} retries")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def search_papers(
        self,
        query: str,
        fields_of_study: Optional[List[str]] = None,
        year: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Optional[Dict]:
        """
        Search for papers by query.
        
        Args:
            query: Search query string
            fields_of_study: Filter by fields (e.g., ['Computer Science', 'Medicine'])
            year: Year range (e.g., '2020-2023')
            limit: Number of results (max 100 per request)
            offset: Pagination offset
            
        Returns:
            API response with paper data
        """
        url = f"{self.BASE_URL}/paper/search"
        
        # Request comprehensive paper fields
        fields = [
            'paperId', 'title', 'abstract', 'year', 'publicationDate',
            'citationCount', 'referenceCount', 'fieldsOfStudy',
            'publicationTypes', 'venue', 'openAccessPdf',
            'authors.authorId', 'authors.name', 'authors.citationCount',
            'authors.hIndex', 'citations.paperId', 'citations.title',
            'references.paperId', 'references.title'
        ]
        
        params = {
            'query': query,
            'fields': ','.join(fields),
            'limit': min(limit, 100),
            'offset': offset
        }
        
        if fields_of_study:
            params['fieldsOfStudy'] = ','.join(fields_of_study)
        if year:
            params['year'] = year
        
        logger.info(f"Searching papers: query='{query}', limit={limit}, offset={offset}")
        return self._make_request(url, params)


class Neo4jBatchInserter:
    """Neo4j database interface with batch insertion support."""
    
    def __init__(self, uri: str, username: str, password: str, batch_size: int = 500):
        """
        Initialize Neo4j connection with batch insertion.
        
        Args:
            uri: Neo4j database URI (e.g., 'bolt://localhost:7687')
            username: Database username
            password: Database password
            batch_size: Number of nodes to batch before insertion
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.batch_size = batch_size
        
        # Batch buffers
        self.paper_batch: List[Dict] = []
        self.author_batch: List[Dict] = []
        self.wrote_batch: List[Dict] = []
        self.cites_batch: List[Dict] = []
        
        self._create_constraints()
        logger.info(f"Neo4j batch inserter initialized with batch_size={batch_size}")
    
    def close(self):
        """Flush remaining batches and close database connection."""
        self.flush_all_batches()
        self.driver.close()
    
    def _create_constraints(self):
        """Create uniqueness constraints and indexes."""
        with self.driver.session() as session:
            # Unique constraints
            session.run(
                "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                "FOR (p:Paper) REQUIRE p.paperId IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT author_id IF NOT EXISTS "
                "FOR (a:Author) REQUIRE a.authorId IS UNIQUE"
            )
            
            # Indexes for performance
            session.run(
                "CREATE INDEX paper_title IF NOT EXISTS "
                "FOR (p:Paper) ON (p.title)"
            )
            session.run(
                "CREATE INDEX author_name IF NOT EXISTS "
                "FOR (a:Author) ON (a.name)"
            )
            session.run(
                "CREATE INDEX paper_year IF NOT EXISTS "
                "FOR (p:Paper) ON (p.year)"
            )
            session.run(
                "CREATE INDEX paper_fields IF NOT EXISTS "
                "FOR (p:Paper) ON (p.fieldsOfStudy)"
            )
            
            logger.info("Neo4j constraints and indexes created")
    
    def add_paper(self, paper: Dict):
        """
        Add paper to batch buffer.
        
        Args:
            paper: Paper data from S2 API
        """
        # Skip if paperId is None
        if not paper.get('paperId'):
            return
            
        paper_data = {
            'paperId': paper.get('paperId'),
            'title': paper.get('title'),
            'abstract': paper.get('abstract'),
            'year': paper.get('year'),
            'publicationDate': paper.get('publicationDate'),
            'citationCount': paper.get('citationCount', 0),
            'referenceCount': paper.get('referenceCount', 0),
            'fieldsOfStudy': paper.get('fieldsOfStudy') or [],
            'publicationTypes': paper.get('publicationTypes') or [],
            'venue': paper.get('venue'),
            'openAccessPdf': paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else None
        }
        
        self.paper_batch.append(paper_data)
        
        if len(self.paper_batch) >= self.batch_size:
            self.flush_paper_batch()
    
    def add_author(self, author: Dict):
        """
        Add author to batch buffer.
        
        Args:
            author: Author data from S2 API
        """
        # Skip if authorId is None
        if not author.get('authorId'):
            return
            
        author_data = {
            'authorId': author.get('authorId'),
            'name': author.get('name'),
            'citationCount': author.get('citationCount', 0),
            'hIndex': author.get('hIndex', 0)
        }
        
        self.author_batch.append(author_data)
        
        if len(self.author_batch) >= self.batch_size:
            self.flush_author_batch()
    
    def add_wrote_relationship(self, author_id: str, paper_id: str):
        """
        Add WROTE relationship to batch buffer.
        
        Args:
            author_id: Author's ID
            paper_id: Paper's ID
        """
        if not author_id or not paper_id:
            return
            
        self.wrote_batch.append({
            'authorId': author_id,
            'paperId': paper_id
        })
        
        if len(self.wrote_batch) >= self.batch_size:
            self.flush_wrote_batch()
    
    def add_cites_relationship(self, citing_paper_id: str, cited_paper_id: str):
        """
        Add CITES relationship to batch buffer.
        
        Args:
            citing_paper_id: ID of paper that cites
            cited_paper_id: ID of paper being cited
        """
        if not citing_paper_id or not cited_paper_id:
            return
            
        self.cites_batch.append({
            'citingPaperId': citing_paper_id,
            'citedPaperId': cited_paper_id
        })
        
        if len(self.cites_batch) >= self.batch_size:
            self.flush_cites_batch()
    
    def flush_paper_batch(self):
        """Flush paper batch to Neo4j using UNWIND."""
        if not self.paper_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS paper
                MERGE (p:Paper {paperId: paper.paperId})
                SET p.title = paper.title,
                    p.abstract = paper.abstract,
                    p.year = paper.year,
                    p.publicationDate = paper.publicationDate,
                    p.citationCount = paper.citationCount,
                    p.referenceCount = paper.referenceCount,
                    p.fieldsOfStudy = paper.fieldsOfStudy,
                    p.publicationTypes = paper.publicationTypes,
                    p.venue = paper.venue,
                    p.openAccessPdf = paper.openAccessPdf,
                    p.lastUpdated = datetime()
                """
                
                session.run(query, batch=self.paper_batch)
                logger.debug(f"Flushed {len(self.paper_batch)} papers to Neo4j")
                self.paper_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing paper batch: {e}")
            self.paper_batch.clear()  # Clear to prevent infinite retry
    
    def flush_author_batch(self):
        """Flush author batch to Neo4j using UNWIND."""
        if not self.author_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS author
                MERGE (a:Author {authorId: author.authorId})
                SET a.name = author.name,
                    a.citationCount = author.citationCount,
                    a.hIndex = author.hIndex,
                    a.lastUpdated = datetime()
                """
                
                session.run(query, batch=self.author_batch)
                logger.debug(f"Flushed {len(self.author_batch)} authors to Neo4j")
                self.author_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing author batch: {e}")
            self.author_batch.clear()
    
    def flush_wrote_batch(self):
        """Flush WROTE relationships to Neo4j using UNWIND."""
        if not self.wrote_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS rel
                MATCH (a:Author {authorId: rel.authorId})
                MATCH (p:Paper {paperId: rel.paperId})
                MERGE (a)-[:WROTE]->(p)
                """
                
                session.run(query, batch=self.wrote_batch)
                logger.debug(f"Flushed {len(self.wrote_batch)} WROTE relationships to Neo4j")
                self.wrote_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing WROTE batch: {e}")
            self.wrote_batch.clear()
    
    def flush_cites_batch(self):
        """Flush CITES relationships to Neo4j using UNWIND."""
        if not self.cites_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS rel
                MATCH (citing:Paper {paperId: rel.citingPaperId})
                MATCH (cited:Paper {paperId: rel.citedPaperId})
                MERGE (citing)-[:CITES]->(cited)
                """
                
                session.run(query, batch=self.cites_batch)
                logger.debug(f"Flushed {len(self.cites_batch)} CITES relationships to Neo4j")
                self.cites_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing CITES batch: {e}")
            self.cites_batch.clear()
    
    def flush_all_batches(self):
        """Flush all pending batches."""
        self.flush_paper_batch()
        self.flush_author_batch()
        self.flush_wrote_batch()
        self.flush_cites_batch()
        logger.info("All batches flushed to Neo4j")
    
    def get_paper_count(self) -> int:
        """Get total number of papers in database."""
        with self.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as count")
            return result.single()['count']
    
    def get_author_count(self) -> int:
        """Get total number of authors in database."""
        with self.driver.session() as session:
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            return result.single()['count']


class ResearchPaperCrawler:
    """Main crawler orchestrating API calls and Neo4j batch storage."""
    
    def __init__(
        self,
        s2_api_key: Optional[str],
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        batch_size: int = 500,
        requests_per_second: float = 0.9
    ):
        """
        Initialize crawler.
        
        Args:
            s2_api_key: Semantic Scholar API key
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            batch_size: Batch size for Neo4j insertions
            requests_per_second: API rate limit
        """
        self.api = SemanticScholarAPI(s2_api_key, requests_per_second)
        self.db = Neo4jBatchInserter(neo4j_uri, neo4j_username, neo4j_password, batch_size)
        self.processed_papers: Set[str] = set()
        self.processed_authors: Set[str] = set()
    
    def crawl_field(
        self,
        field_query: str,
        fields_of_study: Optional[List[str]] = None,
        year_range: Optional[str] = None,
        max_papers: int = 500,
        crawl_citations: bool = True,
        crawl_references: bool = True
    ):
        """
        Crawl papers in a specific research field.
        
        Args:
            field_query: Search query for the field
            fields_of_study: List of fields to filter by
            year_range: Year range (e.g., '2020-2025')
            max_papers: Maximum papers to crawl
            crawl_citations: Whether to crawl citing papers
            crawl_references: Whether to crawl referenced papers
        """
        logger.info(f"Starting crawl for field: {field_query}")
        logger.info(f"Max papers: {max_papers}, Year: {year_range}")
        
        papers_crawled = 0
        offset = 0
        batch_size = 100
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while papers_crawled < max_papers:
            # Search for papers
            response = self.api.search_papers(
                query=field_query,
                fields_of_study=fields_of_study,
                year=year_range,
                limit=batch_size,
                offset=offset
            )
            
            if not response or 'data' not in response:
                consecutive_failures += 1
                logger.warning(f"No data returned. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures. Skipping field: {field_query}")
                    break
                
                # Wait before retrying
                time.sleep(5)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            papers = response['data']
            if not papers:
                logger.info("No more papers available")
                break
            
            # Process each paper
            for paper in papers:
                if papers_crawled >= max_papers:
                    break
                
                try:
                    self._process_paper(
                        paper,
                        crawl_citations=crawl_citations,
                        crawl_references=crawl_references
                    )
                    papers_crawled += 1
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('paperId', 'unknown')}: {e}")
                    continue
            
            # Flush batches periodically
            if papers_crawled % 100 == 0:
                self.db.flush_all_batches()
            
            offset += batch_size
            
            logger.info(f"Progress: {papers_crawled}/{max_papers} papers crawled for '{field_query}'")
        
        # Final flush
        self.db.flush_all_batches()
        
        logger.info(f"Crawl complete for '{field_query}'. Papers crawled: {papers_crawled}")
    
    def _process_paper(
        self,
        paper: Dict,
        crawl_citations: bool = True,
        crawl_references: bool = True
    ):
        """Process a single paper and its relationships."""
        paper_id = paper.get('paperId')
        
        if not paper_id or paper_id in self.processed_papers:
            return
        
        # Add paper to batch
        self.db.add_paper(paper)
        self.processed_papers.add(paper_id)
        
        # Process authors (with null safety)
        authors = paper.get('authors') or []
        for author in authors:
            author_id = author.get('authorId')
            if author_id and author_id not in self.processed_authors:
                self.db.add_author(author)
                self.processed_authors.add(author_id)
            
            if author_id:
                self.db.add_wrote_relationship(author_id, paper_id)
        
        # Process citations (papers that cite this paper) - with null safety
        if crawl_citations:
            citations = paper.get('citations') or []
            for citation in citations[:50]:  # Limit to avoid overwhelming
                cited_paper_id = citation.get('paperId')
                if cited_paper_id:
                    # Add minimal paper node for citation
                    self.db.add_paper({
                        'paperId': cited_paper_id,
                        'title': citation.get('title')
                    })
                    self.db.add_cites_relationship(cited_paper_id, paper_id)
        
        # Process references (papers this paper cites) - with null safety
        if crawl_references:
            references = paper.get('references') or []
            for reference in references[:50]:  # Limit to avoid overwhelming
                ref_paper_id = reference.get('paperId')
                if ref_paper_id:
                    # Add minimal paper node for reference
                    self.db.add_paper({
                        'paperId': ref_paper_id,
                        'title': reference.get('title')
                    })
                    self.db.add_cites_relationship(paper_id, ref_paper_id)
        
        logger.debug(f"Processed paper: {paper.get('title', 'Unknown')[:50]}...")
    
    def crawl_multiple_fields(
        self,
        field_configs: List[Dict],
        max_papers_per_field: int = 200
    ):
        """
        Crawl multiple research fields.
        
        Args:
            field_configs: List of field configurations
            max_papers_per_field: Max papers per field
        """
        total_fields = len(field_configs)
        successful_crawls = 0
        failed_crawls = 0
        
        for i, config in enumerate(field_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Crawling field {i+1}/{total_fields}: {config['query']}")
            logger.info(f"{'='*60}\n")
            
            try:
                self.crawl_field(
                    field_query=config['query'],
                    fields_of_study=config.get('fields'),
                    year_range=config.get('year_range'),
                    max_papers=max_papers_per_field,
                    crawl_citations=config.get('crawl_citations', True),
                    crawl_references=config.get('crawl_references', True)
                )
                successful_crawls += 1
            except Exception as e:
                logger.error(f"Failed to crawl field '{config['query']}': {e}")
                failed_crawls += 1
                # Continue with next field
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-FIELD CRAWL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total fields attempted: {total_fields}")
        logger.info(f"Successful: {successful_crawls}")
        logger.info(f"Failed: {failed_crawls}")
        logger.info(f"{'='*60}")
    
    def close(self):
        """Close all connections and flush remaining batches."""
        self.db.close()


def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Research Paper Crawler for Semantic Scholar API with Neo4j batch insertion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables for API key (recommended)
  export S2_API_KEY="your_api_key_here"
  %(prog)s --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password mypass
  
  # Using command-line API key (less secure)
  %(prog)s --api-key YOUR_KEY --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password mypass
  
  # Using config file
  %(prog)s --config config.json
  
  # With custom batch size and rate limit
  %(prog)s --batch-size 1000 --rate-limit 2.0 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password mypass
        """
    )
    
    # API Configuration
    api_group = parser.add_argument_group('Semantic Scholar API')
    api_group.add_argument(
        '--api-key',
        type=str,
        help='Semantic Scholar API key (alternatively set S2_API_KEY env variable)'
    )
    api_group.add_argument(
        '--rate-limit',
        type=float,
        default=0.9,
        help='API requests per second (default: 0.9 for safety margin under 1 RPS limit)'
    )
    
    # Neo4j Configuration
    neo4j_group = parser.add_argument_group('Neo4j Database')
    neo4j_group.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j database URI (default: bolt://localhost:7687)'
    )
    neo4j_group.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    neo4j_group.add_argument(
        '--neo4j-password',
        type=str,
        help='Neo4j password (alternatively set NEO4J_PASSWORD env variable)'
    )
    
    # Batch Configuration
    batch_group = parser.add_argument_group('Batch Processing')
    batch_group.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Number of nodes to batch before insertion (default: 500)'
    )
    
    # Crawl Configuration
    crawl_group = parser.add_argument_group('Crawl Settings')
    crawl_group.add_argument(
        '--query',
        type=str,
        help='Single search query to crawl'
    )
    crawl_group.add_argument(
        '--fields',
        type=str,
        nargs='+',
        help='Fields of study filter (e.g., "Computer Science" "Medicine")'
    )
    crawl_group.add_argument(
        '--year-range',
        type=str,
        help='Year range filter (e.g., "2020-2025")'
    )
    crawl_group.add_argument(
        '--max-papers',
        type=int,
        default=500,
        help='Maximum papers to crawl per query (default: 500)'
    )
    crawl_group.add_argument(
        '--no-citations',
        action='store_true',
        help='Skip crawling citations'
    )
    crawl_group.add_argument(
        '--no-references',
        action='store_true',
        help='Skip crawling references'
    )
    
    # Config File
    parser.add_argument(
        '--config',
        type=str,
        help='Load configuration from JSON file'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point with argument parsing."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config from file if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
        
        # Override with command-line args if provided
        s2_api_key = args.api_key or config.get('s2_api_key') or os.getenv('S2_API_KEY')
        neo4j_uri = args.neo4j_uri if args.neo4j_uri != 'bolt://localhost:7687' else config.get('neo4j_uri', 'bolt://localhost:7687')
        neo4j_user = args.neo4j_user if args.neo4j_user != 'neo4j' else config.get('neo4j_username', 'neo4j')
        neo4j_password = args.neo4j_password or config.get('neo4j_password') or os.getenv('NEO4J_PASSWORD')
        batch_size = args.batch_size if args.batch_size != 500 else config.get('batch_size', 500)
        rate_limit = args.rate_limit if args.rate_limit != 0.9 else config.get('rate_limit', 0.9)
        field_configs = config.get('field_configs', [])
        max_papers_per_field = config.get('max_papers_per_field', 500)
    else:
        # Get configuration from environment variables and command-line args
        s2_api_key = args.api_key or os.getenv('S2_API_KEY')
        neo4j_uri = args.neo4j_uri
        neo4j_user = args.neo4j_user
        neo4j_password = args.neo4j_password or os.getenv('NEO4J_PASSWORD')
        batch_size = args.batch_size
        rate_limit = args.rate_limit
        field_configs = []
        max_papers_per_field = args.max_papers
        
        # Single query from command line
        if args.query:
            field_configs = [{
                'query': args.query,
                'fields': args.fields,
                'year_range': args.year_range,
                'crawl_citations': not args.no_citations,
                'crawl_references': not args.no_references
            }]
    
    # Validate required parameters
    if not neo4j_password:
        logger.error("Neo4j password not provided. Use --neo4j-password or set NEO4J_PASSWORD environment variable.")
        sys.exit(1)
    
    if not field_configs:
        logger.error("No queries specified. Use --query or provide --config file.")
        sys.exit(1)
    
    # Display configuration
    logger.info("="*60)
    logger.info("CRAWLER CONFIGURATION")
    logger.info("="*60)
    logger.info(f"API Key: {'Configured' if s2_api_key else 'Not provided (unauthenticated)'}")
    logger.info(f"Rate Limit: {rate_limit} requests/second")
    logger.info(f"Neo4j URI: {neo4j_uri}")
    logger.info(f"Neo4j User: {neo4j_user}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Number of Queries: {len(field_configs)}")
    logger.info(f"Max Papers per Query: {max_papers_per_field}")
    logger.info("="*60)
    
    # Initialize crawler
    crawler = ResearchPaperCrawler(
        s2_api_key=s2_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_user,
        neo4j_password=neo4j_password,
        batch_size=batch_size,
        requests_per_second=rate_limit
    )
    
    try:
        # Run crawler
        crawler.crawl_multiple_fields(
            field_configs=field_configs,
            max_papers_per_field=max_papers_per_field
        )
        
        # Print statistics
        print("\n" + "="*60)
        print("CRAWL STATISTICS")
        print("="*60)
        print(f"Total Papers: {crawler.db.get_paper_count()}")
        print(f"Total Authors: {crawler.db.get_author_count()}")
        print(f"Processed Papers: {len(crawler.processed_papers)}")
        print(f"Processed Authors: {len(crawler.processed_authors)}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\nCrawl interrupted by user. Flushing remaining batches...")
    except Exception as e:
        logger.error(f"Crawler error: {e}", exc_info=True)
    finally:
        crawler.close()
        logger.info("Crawler shutdown complete")


if __name__ == "__main__":
    main()
