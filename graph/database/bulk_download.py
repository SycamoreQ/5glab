import requests
import json
import gzip
import os
from typing import Dict, Optional, List, Set
from neo4j import GraphDatabase
import logging
from tqdm import tqdm
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticScholarBulkLoader:
    """Download and load Semantic Scholar bulk datasets into Neo4j."""
    
    DATASETS_BASE_URL = "https://api.semanticscholar.org/datasets/v1"
    
    def __init__(
        self,
        api_key: str,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        batch_size: int = 1000,
        download_dir: str = "/Users/kaushikmuthukumar/Downloads/s2_datasets"
    ):
        """
        Initialize bulk loader.
        
        Args:
            api_key: Semantic Scholar API key
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            batch_size: Batch size for Neo4j insertions
            download_dir: Directory to store downloaded files
        """
        self.api_key = api_key
        self.download_dir = download_dir
        self.batch_size = batch_size
        
        # Neo4j connection
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        self._create_constraints()
        
        # Batch buffers
        self.paper_batch: List[Dict] = []
        self.author_batch: List[Dict] = []
        self.wrote_batch: List[Dict] = []
        self.cites_batch: List[Dict] = []
        
        # Cache for duplicate checking
        self.existing_papers_cache: Set[str] = set()
        self.existing_authors_cache: Set[str] = set()
        self.cache_loaded = False
        self.cache_check_interval = 50000  # Refresh cache every 50k papers
        
        # Statistics
        self.stats = {
            'papers_processed': 0,
            'authors_processed': 0,
            'citations_processed': 0,
            'papers_inserted': 0,
            'authors_inserted': 0,
            'duplicates_skipped': 0
        }
        
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Bulk loader initialized. Download dir: {download_dir}")
    
    def _create_constraints(self):
        """Create Neo4j constraints and indexes."""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                    "FOR (p:Paper) REQUIRE p.paperId IS UNIQUE"
                )
                session.run(
                    "CREATE CONSTRAINT author_id IF NOT EXISTS "
                    "FOR (a:Author) REQUIRE a.authorId IS UNIQUE"
                )
                session.run("CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)")
                session.run("CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)")
                session.run("CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)")
                session.run("CREATE INDEX paper_fields IF NOT EXISTS FOR (p:Paper) ON (p.fieldsOfStudy)")
                
                logger.info("Neo4j constraints and indexes created")
            except Exception as e:
                logger.warning(f"Constraint creation warning (may already exist): {e}")
    
    def load_existing_papers_cache(self):
        """Load existing paper IDs from Neo4j into memory cache."""
        logger.info("Loading existing paper IDs from Neo4j...")
        
        with self.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN p.paperId as paperId")
            
            count = 0
            for record in result:
                self.existing_papers_cache.add(record['paperId'])
                count += 1
                
                if count % 100000 == 0:
                    logger.info(f"Loaded {count:,} paper IDs into cache...")
        
        self.cache_loaded = True
        logger.info(f"Cache loaded with {len(self.existing_papers_cache):,} existing paper IDs")
    
    def load_existing_authors_cache(self):
        """Load existing author IDs from Neo4j into memory cache."""
        logger.info("Loading existing author IDs from Neo4j...")
        
        with self.driver.session() as session:
            result = session.run("MATCH (a:Author) RETURN a.authorId as authorId")
            
            for record in result:
                self.existing_authors_cache.add(record['authorId'])
        
        logger.info(f"Loaded {len(self.existing_authors_cache):,} existing author IDs")
    
    def refresh_cache_if_needed(self):
        """Refresh cache periodically during processing."""
        if self.stats['papers_processed'] % self.cache_check_interval == 0 and self.stats['papers_processed'] > 0:
            logger.info("Refreshing existing papers cache...")
            old_size = len(self.existing_papers_cache)
            self.load_existing_papers_cache()
            new_size = len(self.existing_papers_cache)
            logger.info(f"Cache refreshed: {old_size:,} -> {new_size:,} papers ({new_size - old_size:,} new)")
    
    def get_latest_release(self) -> str:
        """Get the latest dataset release ID."""
        try:
            response = requests.get(f"{self.DATASETS_BASE_URL}/release/latest")
            response.raise_for_status()
            release_data = response.json()
            release_id = release_data['release_id']
            logger.info(f"Latest release: {release_id}")
            return release_id
        except Exception as e:
            logger.error(f"Failed to get latest release: {e}")
            raise
    
    def get_available_datasets(self, release_id: str) -> List[str]:
        """Get list of available datasets in a release."""
        try:
            response = requests.get(f"{self.DATASETS_BASE_URL}/release/{release_id}")
            response.raise_for_status()
            datasets = response.json()
            
            dataset_names = [d['name'] for d in datasets['datasets']]
            logger.info(f"Available datasets: {dataset_names}")
            return dataset_names
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            raise
    
    def get_download_urls(self, release_id: str, dataset_name: str) -> Dict:
        """Get download URLs for a specific dataset."""
        headers = {'x-api-key': self.api_key} if self.api_key else {}
        
        try:
            response = requests.get(
                f"{self.DATASETS_BASE_URL}/release/{release_id}/dataset/{dataset_name}",
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Files available: {len(data['files'])}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to get download URLs: {e}")
            raise
    
    def download_dataset(self, release_id: str, dataset_name: str, max_files: Optional[int] = None):
        """
        Download dataset files.
        
        Args:
            release_id: Release ID to download
            dataset_name: Name of dataset ('papers', 'citations', etc.)
            max_files: Maximum number of files to download (None = all)
        """
        download_info = self.get_download_urls(release_id, dataset_name)
        
        files = download_info['files']
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Downloading {len(files)} files for {dataset_name}...")
        
        for i, url in enumerate(files):
            filename = os.path.basename(url.split('?')[0])  # Remove query params
            filepath = os.path.join(self.download_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                logger.info(f"File {i+1}/{len(files)} already exists: {filename}")
                continue
            
            try:
                logger.info(f"Downloading {i+1}/{len(files)}: {filename}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
    
    def _process_paper(self, paper: Dict):
        """Process a single paper record with duplicate checking."""
        # FIXED: Use lowercase field names (actual format in dataset)
        corpus_id = paper.get('corpusid')
        if not corpus_id:
            return
        
        paper_id = str(corpus_id)
        
        # Check if paper already exists in cache
        if self.cache_loaded and paper_id in self.existing_papers_cache:
            self.stats['duplicates_skipped'] += 1
            return
        
        # Prepare paper data - FIXED: use lowercase keys
        paper_data = {
            'paperId': paper_id,
            'externalIds': json.dumps(paper.get('externalids', {})),
            'title': paper.get('title'),
            'abstract': paper.get('abstract'),
            'year': paper.get('year'),
            'publicationDate': paper.get('publicationdate'),
            'citationCount': paper.get('citationcount', 0),
            'referenceCount': paper.get('referencecount', 0),
            'fieldsOfStudy': paper.get('fieldsofstudy') or [],
            'publicationTypes': paper.get('publicationtypes') or [],
            'venue': paper.get('venue'),
            'openAccessPdf': paper.get('openaccesspdf', {}).get('url') if paper.get('openaccesspdf') else None,
            's2FieldsOfStudy': json.dumps(paper.get('s2fieldsofstudy', [])),
            'publicationVenue': json.dumps(paper.get('publicationvenue', {}))
        }
        
        self.paper_batch.append(paper_data)
        self.existing_papers_cache.add(paper_id)
        
        # Process authors - FIXED: lowercase 'authorid'
        authors = paper.get('authors', []) or []
        for author in authors:
            author_id = author.get('authorid') or author.get('authorId')
            if author_id:
                author_id_str = str(author_id)
                
                # Only add author if not in cache
                if author_id_str not in self.existing_authors_cache:
                    author_data = {
                        'authorId': author_id_str,
                        'name': author.get('name')
                    }
                    self.author_batch.append(author_data)
                    self.existing_authors_cache.add(author_id_str)
                
                # WROTE relationship
                self.wrote_batch.append({
                    'authorId': author_id_str,
                    'paperId': paper_id
                })
        
        # Process references - FIXED: lowercase
        references = paper.get('references', []) or []
        for reference in references[:100]:
            ref_id = reference.get('corpusid')
            if ref_id:
                ref_id_str = str(ref_id)
                
                if ref_id_str not in self.existing_papers_cache:
                    ref_paper_data = {
                        'paperId': ref_id_str,
                        'title': reference.get('title')
                    }
                    self.paper_batch.append(ref_paper_data)
                    self.existing_papers_cache.add(ref_id_str)
                
                self.cites_batch.append({
                    'citingPaperId': paper_id,
                    'citedPaperId': ref_id_str
                })
        
        # Flush if batch size reached
        if len(self.paper_batch) >= self.batch_size:
            self.flush_paper_batch()
        if len(self.author_batch) >= self.batch_size:
            self.flush_author_batch()
        if len(self.wrote_batch) >= self.batch_size:
            self.flush_wrote_batch()
        if len(self.cites_batch) >= self.batch_size:
            self.flush_cites_batch()
    
    def flush_paper_batch(self):
        """Flush papers to Neo4j."""
        if not self.paper_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS paper
                MERGE (p:Paper {paperId: paper.paperId})
                ON CREATE SET
                    p.title = paper.title,
                    p.abstract = paper.abstract,
                    p.year = paper.year,
                    p.publicationDate = paper.publicationDate,
                    p.citationCount = paper.citationCount,
                    p.referenceCount = paper.referenceCount,
                    p.fieldsOfStudy = paper.fieldsOfStudy,
                    p.publicationTypes = paper.publicationTypes,
                    p.venue = paper.venue,
                    p.openAccessPdf = paper.openAccessPdf,
                    p.externalIds = paper.externalIds,
                    p.s2FieldsOfStudy = paper.s2FieldsOfStudy,
                    p.publicationVenue = paper.publicationVenue,
                    p.createdAt = datetime()
                ON MATCH SET
                    p.lastUpdated = datetime(),
                    p.citationCount = CASE WHEN paper.citationCount IS NOT NULL 
                                           THEN paper.citationCount 
                                           ELSE p.citationCount END
                """
                
                session.run(query, batch=self.paper_batch)
                self.stats['papers_inserted'] += len(self.paper_batch)
                self.paper_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing paper batch: {e}")
            self.paper_batch.clear()
    
    def flush_author_batch(self):
        """Flush authors to Neo4j."""
        if not self.author_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS author
                MERGE (a:Author {authorId: author.authorId})
                ON CREATE SET
                    a.name = author.name,
                    a.createdAt = datetime()
                ON MATCH SET
                    a.lastUpdated = datetime()
                """
                
                session.run(query, batch=self.author_batch)
                self.stats['authors_inserted'] += len(self.author_batch)
                self.author_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing author batch: {e}")
            self.author_batch.clear()
    
    def flush_wrote_batch(self):
        """Flush WROTE relationships to Neo4j."""
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
                self.wrote_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing WROTE batch: {e}")
            self.wrote_batch.clear()
    
    def flush_cites_batch(self):
        """Flush CITES relationships to Neo4j."""
        if not self.cites_batch:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $batch AS rel
                MERGE (citing:Paper {paperId: rel.citingPaperId})
                MERGE (cited:Paper {paperId: rel.citedPaperId})
                MERGE (citing)-[:CITES]->(cited)
                """
                
                session.run(query, batch=self.cites_batch)
                self.stats['citations_processed'] += len(self.cites_batch)
                self.cites_batch.clear()
        except Exception as e:
            logger.error(f"Error flushing CITES batch: {e}")
            self.cites_batch.clear()
    
    def flush_all_batches(self):
        """Flush all batches."""
        self.flush_paper_batch()
        self.flush_author_batch()
        self.flush_wrote_batch()
        self.flush_cites_batch()
    
    def process_papers_file(
        self,
        filepath: str,
        filter_fields: Optional[List[str]] = None,
        filter_year_min: Optional[int] = None,
        filter_year_max: Optional[int] = None
    ):
        """
        Process a papers JSONL.gz file and load into Neo4j.
        
        Args:
            filepath: Path to the .gz file
            filter_fields: Only process papers with these fields of study
            filter_year_min: Minimum year filter
            filter_year_max: Maximum year filter
        """
        logger.info(f"Processing file: {filepath}")
        
        # Load existing papers cache on first file
        if not self.cache_loaded:
            self.load_existing_papers_cache()
            self.load_existing_authors_cache()
        
        initial_duplicates = self.stats['duplicates_skipped']
        
        # Diagnostic counters
        total_lines = 0
        filtered_by_year = 0
        filtered_by_field = 0
        missing_corpus_id = 0
        sample_papers_shown = 0
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        paper = json.loads(line)
                        total_lines += 1
                        
                        # Show first 3 papers for debugging
                        if sample_papers_shown < 3:
                            logger.info(f"\n=== SAMPLE PAPER {sample_papers_shown + 1} ===")
                            logger.info(f"Title: {paper.get('title', 'N/A')[:100]}")
                            logger.info(f"Year: {paper.get('year')}")
                            logger.info(f"S2 Fields of Study: {paper.get('s2fieldsofstudy')}")
                            logger.info(f"Corpus ID: {paper.get('corpusid')}")
                            sample_papers_shown += 1
                        
                        # FIXED: Check for corpus ID (lowercase)
                        corpus_id = paper.get('corpusid')
                        if not corpus_id:
                            missing_corpus_id += 1
                            continue
                        
                        # Check for duplicate before filtering (faster)
                        if str(corpus_id) in self.existing_papers_cache:
                            self.stats['duplicates_skipped'] += 1
                            continue
                        
                        # Apply year filters
                        paper_year = paper.get('year')
                        if filter_year_min and paper_year:
                            if paper_year < filter_year_min:
                                filtered_by_year += 1
                                continue
                        
                        if filter_year_max and paper_year:
                            if paper_year > filter_year_max:
                                filtered_by_year += 1
                                continue
                        
                        # FIXED: Apply field filters using s2fieldsofstudy
                        if filter_fields:
                            paper_fields = paper.get('s2fieldsofstudy', []) or []
                            
                            # Extract 'category' from s2fieldsofstudy objects
                            # Format: [{"category": "Computer Science", "source": "s2-fos-model"}]
                            field_names = []
                            if isinstance(paper_fields, list):
                                for field in paper_fields:
                                    if isinstance(field, dict):
                                        field_names.append(field.get('category'))
                                    elif isinstance(field, str):
                                        field_names.append(field)
                            
                            if not any(field in field_names for field in filter_fields):
                                filtered_by_field += 1
                                continue
                        
                        # Process paper
                        self._process_paper(paper)
                        self.stats['papers_processed'] += 1
                        
                        # Periodic cache refresh
                        self.refresh_cache_if_needed()
                        
                        # Periodic flush and logging
                        if self.stats['papers_processed'] % 10000 == 0:
                            self.flush_all_batches()
                            logger.info(f"Progress: {self.stats['papers_processed']:,} papers processed")
                            logger.info(f"Duplicates skipped: {self.stats['duplicates_skipped']:,}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        continue
            
            # Final flush
            self.flush_all_batches()
            
            duplicates_in_file = self.stats['duplicates_skipped'] - initial_duplicates
            
            # Diagnostic summary
            logger.info(f"\n{'='*60}")
            logger.info(f"FILE PROCESSING SUMMARY: {os.path.basename(filepath)}")
            logger.info(f"{'='*60}")
            logger.info(f"Total lines read: {total_lines:,}")
            logger.info(f"Missing corpus ID: {missing_corpus_id:,}")
            logger.info(f"Filtered by year: {filtered_by_year:,}")
            logger.info(f"Filtered by field: {filtered_by_field:,}")
            logger.info(f"Duplicates skipped: {duplicates_in_file:,}")
            logger.info(f"Papers processed: {self.stats['papers_processed']:,}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Failed to process file {filepath}: {e}")
    
    def process_citations_file(self, filepath: str):
        """
        Process a citations JSONL.gz file and load CITES relationships into Neo4j.
        
        Args:
            filepath: Path to the .gz citations file
        """
        logger.info(f"Processing citations file: {filepath}")
        
        total_lines = 0
        citations_added = 0
        missing_ids = 0
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        citation = json.loads(line)
                        total_lines += 1
                        
                        # Show first 3 citations for debugging
                        if citations_added < 3:
                            logger.info(f"\n=== SAMPLE CITATION {citations_added + 1} ===")
                            logger.info(f"Citing: {citation.get('citingcorpusid')}")
                            logger.info(f"Cited: {citation.get('citedcorpusid')}")
                        
                        # Get citation pair (lowercase keys)
                        citing_id = citation.get('citingcorpusid')
                        cited_id = citation.get('citedcorpusid')
                        
                        if not citing_id or not cited_id:
                            missing_ids += 1
                            continue
                        
                        # Add to batch
                        self.cites_batch.append({
                            'citingPaperId': str(citing_id),
                            'citedPaperId': str(cited_id)
                        })
                        
                        citations_added += 1
                        
                        # Flush if batch size reached
                        if len(self.cites_batch) >= self.batch_size:
                            self.flush_cites_batch()
                        
                        # Periodic logging
                        if citations_added % 100000 == 0:
                            self.flush_cites_batch()
                            logger.info(f"Progress: {citations_added:,} citations processed")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        continue
            
            # Final flush
            self.flush_cites_batch()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"CITATIONS FILE SUMMARY: {os.path.basename(filepath)}")
            logger.info(f"{'='*60}")
            logger.info(f"Total lines read: {total_lines:,}")
            logger.info(f"Missing IDs: {missing_ids:,}")
            logger.info(f"Citations added: {citations_added:,}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Failed to process citations file {filepath}: {e}")
    
    def process_all_papers(
        self,
        filter_fields: Optional[List[str]] = None,
        filter_year_min: int = 2018,
        filter_year_max: int = 2025,
        max_files: Optional[int] = None
    ):
        """
        Process all downloaded papers files.
        
        Args:
            filter_fields: Filter by fields of study
            filter_year_min: Minimum year
            filter_year_max: Maximum year
            max_files: Maximum number of files to process
        """
        # FIXED: Look for .gz files (not .jsonl.gz)
        files = sorted([f for f in os.listdir(self.download_dir) if f.endswith('.gz')])
        
        if not files:
            logger.error(f"No .gz files found in {self.download_dir}")
            return
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} files to process")
        logger.info(f"Filters: fields={filter_fields}, years={filter_year_min}-{filter_year_max}")
        
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(self.download_dir, filename)
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {i}/{len(files)}: {filename}")
            logger.info(f"{'='*60}")
            
            self.process_papers_file(
                filepath,
                filter_fields=filter_fields,
                filter_year_min=filter_year_min,
                filter_year_max=filter_year_max
            )
        
        self.print_statistics()
    
    def process_all_citations(self, max_files: Optional[int] = None):
        """
        Process all downloaded citation files.
        
        Args:
            max_files: Maximum number of files to process
        """
        files = sorted([f for f in os.listdir(self.download_dir) if f.endswith('.gz')])
        
        if not files:
            logger.error(f"No .gz files found in {self.download_dir}")
            return
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} citation files to process")
        
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(self.download_dir, filename)
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing citation file {i}/{len(files)}: {filename}")
            logger.info(f"{'='*60}")
            
            self.process_citations_file(filepath)
        
        self.print_statistics()
    
    def print_statistics(self):
        """Print loading statistics."""
        # Get actual counts from Neo4j
        try:
            with self.driver.session() as session:
                paper_count = session.run("MATCH (p:Paper) RETURN count(p) as count").single()['count']
                author_count = session.run("MATCH (a:Author) RETURN count(a) as count").single()['count']
                cites_count = session.run("MATCH ()-[r:CITES]->() RETURN count(r) as count").single()['count']
                wrote_count = session.run("MATCH ()-[r:WROTE]->() RETURN count(r) as count").single()['count']
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            paper_count = author_count = cites_count = wrote_count = 0
        
        print("\n" + "="*60)
        print("BULK LOAD STATISTICS")
        print("="*60)
        print(f"Papers processed (new): {self.stats['papers_processed']:,}")
        print(f"Duplicates skipped: {self.stats['duplicates_skipped']:,}")
        print(f"Papers inserted (batch): {self.stats['papers_inserted']:,}")
        print(f"Authors inserted (batch): {self.stats['authors_inserted']:,}")
        print(f"Citations processed: {self.stats['citations_processed']:,}")
        print("\n" + "-"*60)
        print("CURRENT DATABASE TOTALS")
        print("-"*60)
        print(f"Total Papers in DB: {paper_count:,}")
        print(f"Total Authors in DB: {author_count:,}")
        print(f"Total CITES relationships: {cites_count:,}")
        print(f"Total WROTE relationships: {wrote_count:,}")
        print("="*60)
    
    def close(self):
        """Close connections."""
        self.flush_all_batches()
        self.driver.close()
        logger.info("Bulk loader closed")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Load Semantic Scholar bulk dataset into Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process papers with filters
  python bulk_loader.py --api-key YOUR_KEY --neo4j-password PASS \\
    --dataset-type papers --filter-fields "Computer Science" "Medicine" --year-min 2020

  # Download and process citations
  python bulk_loader.py --api-key YOUR_KEY --neo4j-password PASS \\
    --dataset-type citations --download-dir ./s2_citations --batch-size 5000

  # Download only (first run)
  python bulk_loader.py --api-key YOUR_KEY --neo4j-password PASS \\
    --download-only --max-files 5

  # Process existing files
  python bulk_loader.py --api-key YOUR_KEY --neo4j-password PASS \\
    --process-only --filter-fields "Computer Science" --year-min 2018
        """
    )
    
    # Required arguments
    parser.add_argument('--api-key', required=True, help='Semantic Scholar API key')
    parser.add_argument('--neo4j-password', required=True, help='Neo4j password')
    
    # Optional Neo4j configuration
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    
    # Processing configuration
    parser.add_argument('--download-dir', default='/Users/kaushikmuthukumar/Downloads/s2_datasets', 
                       help='Download directory')
    parser.add_argument('--batch-size', type=int, default=500000, help='Neo4j batch size')
    parser.add_argument('--max-files', type=int, help='Max files to download/process')
    
    # Dataset type
    parser.add_argument('--dataset-type', default='papers', choices=['papers', 'citations'], 
                       help='Dataset type to download/process')
    
    # Operation mode
    parser.add_argument('--download-only', action='store_true', help='Only download, don\'t process')
    parser.add_argument('--process-only', action='store_true', help='Only process existing files')
    
    # Filters (only for papers)
    parser.add_argument('--filter-fields', nargs='+', help='Filter by fields of study (papers only)')
    parser.add_argument('--year-min', type=int, default=2018, help='Minimum year (papers only)')
    parser.add_argument('--year-max', type=int, default=2025, help='Maximum year (papers only)')
    
    args = parser.parse_args()
    
    # Validation
    if args.dataset_type == 'citations' and (args.filter_fields or args.year_min != 2018 or args.year_max != 2025):
        logger.warning("⚠️  Filters (--filter-fields, --year-min, --year-max) are ignored for citation files!")
    
    logger.info("="*60)
    logger.info("SEMANTIC SCHOLAR BULK LOADER")
    logger.info("="*60)
    
    # Initialize loader
    try:
        loader = SemanticScholarBulkLoader(
            api_key=args.api_key,
            neo4j_uri=args.neo4j_uri,
            neo4j_username=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            batch_size=args.batch_size,
            download_dir=args.download_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize loader: {e}")
        sys.exit(1)
    
    try:
        # FIXED: Only download if not process-only
        if not args.process_only:
            logger.info("\n" + "="*60)
            logger.info("DOWNLOAD PHASE")
            logger.info("="*60)
            release_id = loader.get_latest_release()
            loader.get_available_datasets(release_id)
            loader.download_dataset(release_id, args.dataset_type, max_files=args.max_files)
        
        # FIXED: Only process if not download-only
        if not args.download_only:
            logger.info("\n" + "="*60)
            logger.info("PROCESSING PHASE")
            logger.info("="*60)
            
            if args.dataset_type == 'papers':
                loader.process_all_papers(
                    filter_fields=args.filter_fields,
                    filter_year_min=args.year_min,
                    filter_year_max=args.year_max,
                    max_files=args.max_files
                )
            elif args.dataset_type == 'citations':
                # FIXED: Pass max_files argument
                loader.process_all_citations(max_files=args.max_files)
    
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user. Flushing remaining batches...")
        loader.print_statistics()
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        loader.close()



if __name__ == "__main__":
    main()
