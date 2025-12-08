import asyncio
import aiohttp
import json
import hashlib
from typing import Dict, List, Set, Optional
from tqdm.asyncio import tqdm
from neo4j import AsyncGraphDatabase, exceptions


class HybridDatabaseBuilder:
    
    def __init__(self, s2_api_key: str, database_name: str = "researchdb_hybrid"):
        self.s2_api_key = s2_api_key
        self.database_name = database_name
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Deduplication index
        self.title_to_id = {}
        self.doi_to_id = {}
        self.arxiv_to_id = {}
        self.s2_to_id = {}
        
        # Tracking
        self.pending_citations = []
        self.driver = None
        self.session = None
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(80)
        self.request_count = 0
        self.window_start = None
    
    async def __aenter__(self):
        self.driver = AsyncGraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "diam0ndman@3"),
            max_connection_pool_size=100
        )
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"x-api-key": self.s2_api_key}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.driver:
            await self.driver.close()
    
    def normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        if not title:
            return ""
        title = title.lower().strip()
        title = ''.join(c for c in title if c.isalnum() or c.isspace())
        return ' '.join(title.split())
    
    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI."""
        if not doi:
            return ""
        doi = doi.lower().strip()
        doi = doi.replace('https://doi.org/', '')
        doi = doi.replace('http://dx.doi.org/', '')
        return doi
    
    async def setup_database(self):
        """Create database with indexes."""
        print("="*80)
        print("HYBRID DATABASE BUILDER")
        print("="*80)
        print("Sources: arXiv (2M) + Semantic Scholar (6M)")
        print(f"Database: {self.database_name}")
        print("Total target: 8M papers with unified citation network\n")
        
        # Create database
        async with self.driver.session(database="system") as session:
            try:
                await session.run(f"CREATE DATABASE {self.database_name}")
                print(f"✓ Created database: {self.database_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"✓ Using existing database: {self.database_name}")
                else:
                    raise
        
        # Create indexes
        print("✓ Creating indexes...")
        async with self.driver.session(database=self.database_name) as session:
            indexes = [
                "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
                "CREATE INDEX paper_arxiv IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)",
                "CREATE INDEX paper_s2 IF NOT EXISTS FOR (p:Paper) ON (p.s2_paper_id)",
                "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX paper_source IF NOT EXISTS FOR (p:Paper) ON (p.source)",
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.unified_id IS UNIQUE",
            ]
            for idx in indexes:
                try:
                    await session.run(idx)
                except:
                    pass
        
        print("✓ Database ready!\n")
    
    async def check_existing_data(self) -> Dict[str, int]:
        """
        ✅ NEW: Check what data already exists in database.
        Returns dict with counts of arxiv papers, s2 papers, total papers.
        """
        print("="*80)
        print("CHECKING EXISTING DATA")
        print("="*80)
        
        async with self.driver.session(database=self.database_name) as session:
            # Count arXiv papers
            result = await session.run(
                "MATCH (p:Paper {source: 'arxiv'}) RETURN count(p) as count"
            )
            record = await result.single()
            arxiv_count = record['count'] if record else 0
            
            # Count S2 papers
            result = await session.run(
                "MATCH (p:Paper {source: 's2api'}) RETURN count(p) as count"
            )
            record = await result.single()
            s2_count = record['count'] if record else 0
            
            # Count total papers
            result = await session.run(
                "MATCH (p:Paper) RETURN count(p) as count"
            )
            record = await result.single()
            total_count = record['count'] if record else 0
            
            # Count citations
            result = await session.run(
                "MATCH ()-[r:CITES]->() RETURN count(r) as count"
            )
            record = await result.single()
            citation_count = record['count'] if record else 0
        
        print(f"\nExisting data:")
        print(f"  arXiv papers:  {arxiv_count:,}")
        print(f"  S2 papers:     {s2_count:,}")
        print(f"  Total papers:  {total_count:,}")
        print(f"  Citations:     {citation_count:,}")
        print()
        
        return {
            'arxiv': arxiv_count,
            's2': s2_count,
            'total': total_count,
            'citations': citation_count
        }
    
    async def load_existing_indexes(self):
        """
        ✅ NEW: Load existing papers into deduplication indexes.
        This prevents duplicate detection when resuming.
        """
        print("Loading existing papers into memory index...")
        
        async with self.driver.session(database=self.database_name) as session:
            # Load all existing papers in batches
            offset = 0
            batch_size = 10000
            total_loaded = 0
            
            while True:
                query = """
                MATCH (p:Paper)
                RETURN elementId(p) as neo4j_id, 
                       p.title as title,
                       p.doi as doi,
                       p.arxiv_id as arxiv_id,
                       p.s2_paper_id as s2_id
                SKIP $offset
                LIMIT $limit
                """
                
                result = await session.run(query, offset=offset, limit=batch_size)
                records = await result.data()
                
                if not records:
                    break
                
                for record in records:
                    neo4j_id = record['neo4j_id']
                    
                    # Add to indexes
                    if record['title']:
                        norm_title = self.normalize_title(record['title'])
                        self.title_to_id[norm_title] = neo4j_id
                    
                    if record['doi']:
                        norm_doi = self.normalize_doi(record['doi'])
                        self.doi_to_id[norm_doi] = neo4j_id
                    
                    if record['arxiv_id']:
                        self.arxiv_to_id[record['arxiv_id']] = neo4j_id
                    
                    if record['s2_id']:
                        self.s2_to_id[record['s2_id']] = neo4j_id
                
                total_loaded += len(records)
                offset += batch_size
                
                if total_loaded % 50000 == 0:
                    print(f"  Loaded {total_loaded:,} papers into index...", end='\r')
        
        print(f"✓ Loaded {total_loaded:,} papers into memory index")
        print(f"  Title index:  {len(self.title_to_id):,}")
        print(f"  DOI index:    {len(self.doi_to_id):,}")
        print(f"  arXiv index:  {len(self.arxiv_to_id):,}")
        print(f"  S2 index:     {len(self.s2_to_id):,}")
        print()
    
    async def phase1_import_arxiv(self, arxiv_file: str, max_papers: int = 2000000):
        """
        Phase 1: Import arXiv papers.
        ✅ FIXED: Uses MERGE instead of CREATE to avoid constraint errors.
        """
        print("="*80)
        print("PHASE 1: IMPORTING ARXIV")
        print("="*80)
        print(f"Target: {max_papers:,} papers\n")
        
        added = 0
        skipped = 0
        
        async with self.driver.session(database=self.database_name) as session:
            with open(arxiv_file, 'r', encoding='utf-8') as f:
                batch = []
                batch_size = 100
                
                for line in tqdm(f, desc="Processing arXiv", total=max_papers):
                    if added >= max_papers:
                        break
                    
                    try:
                        paper = json.loads(line)
                    except:
                        continue
                    
                    # Extract
                    title = paper.get('title', '').strip()
                    abstract = paper.get('abstract', '').strip()
                    
                    if not title or not abstract or len(abstract) < 100:
                        continue
                    
                    # Clean
                    abstract = ' '.join(abstract.split())
                    
                    # Check duplicate in memory index
                    norm_title = self.normalize_title(title)
                    if norm_title in self.title_to_id:
                        skipped += 1
                        continue
                    
                    # Prepare
                    arxiv_id = paper.get('id', '')
                    
                    # Check arXiv ID duplicate
                    if arxiv_id and arxiv_id in self.arxiv_to_id:
                        skipped += 1
                        continue
                    
                    year = int(paper.get('update_date', '2000')[:4])
                    doi = self.normalize_doi(paper.get('doi', ''))
                    categories = paper.get('categories', '')
                    authors = paper.get('authors_parsed', [])
                    
                    # Generate unified ID
                    unified_id = f"arxiv_{arxiv_id}" if arxiv_id else f"title_{hashlib.md5(norm_title.encode()).hexdigest()[:16]}"
                    
                    batch.append({
                        'unified_id': unified_id,
                        'title': title,
                        'abstract': abstract,
                        'year': year,
                        'doi': doi,
                        'arxiv_id': arxiv_id,
                        's2_paper_id': '',
                        'keywords': categories,
                        'source': 'arxiv',
                        'authors': [f"{a[0]} {a[1]}".strip() for a in authors if len(a) >= 2]
                    })
                    
                    # Batch insert
                    if len(batch) >= batch_size:
                        neo4j_ids = await self._insert_batch_safe(session, batch)
                        
                        # Update dedup indexes
                        for paper_data, neo4j_id in zip(batch, neo4j_ids):
                            if neo4j_id:  # Only if successfully inserted
                                norm_title = self.normalize_title(paper_data['title'])
                                self.title_to_id[norm_title] = neo4j_id
                                
                                if paper_data['doi']:
                                    self.doi_to_id[paper_data['doi']] = neo4j_id
                                if paper_data['arxiv_id']:
                                    self.arxiv_to_id[paper_data['arxiv_id']] = neo4j_id
                                
                                added += 1
                        
                        batch = []
                
                # Insert remaining
                if batch:
                    neo4j_ids = await self._insert_batch_safe(session, batch)
                    for paper_data, neo4j_id in zip(batch, neo4j_ids):
                        if neo4j_id:
                            norm_title = self.normalize_title(paper_data['title'])
                            self.title_to_id[norm_title] = neo4j_id
                            if paper_data['doi']:
                                self.doi_to_id[paper_data['doi']] = neo4j_id
                            if paper_data['arxiv_id']:
                                self.arxiv_to_id[paper_data['arxiv_id']] = neo4j_id
                            added += 1
        
        print(f"\n✓ Phase 1 complete:")
        print(f"  Added:   {added:,} new arXiv papers")
        print(f"  Skipped: {skipped:,} duplicates")
        print(f"  Total index size: {len(self.title_to_id):,}\n")
        
        return added
    
    async def _insert_batch_safe(self, session, batch: List[Dict]) -> List[Optional[str]]:
        """
        ✅ FIXED: Insert batch using MERGE to avoid constraint errors.
        Returns list of Neo4j IDs (None for duplicates).
        """
        query = """
        UNWIND $batch as paper
        MERGE (p:Paper {unified_id: paper.unified_id})
        ON CREATE SET
            p.title = paper.title,
            p.abstract = paper.abstract,
            p.year = paper.year,
            p.doi = paper.doi,
            p.arxiv_id = paper.arxiv_id,
            p.s2_paper_id = paper.s2_paper_id,
            p.keywords = paper.keywords,
            p.source = paper.source
        WITH p, paper
        UNWIND paper.authors as author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:WROTE]->(p)
        RETURN elementId(p) as neo4j_id
        """
        
        try:
            result = await session.run(query, batch=batch)
            records = await result.data()
            return [r['neo4j_id'] for r in records]
        except exceptions.ConstraintError as e:
            print(f"\n⚠ Batch constraint error, inserting individually...")
            neo4j_ids = []
            for paper_data in batch:
                try:
                    result = await session.run(query, batch=[paper_data])
                    records = await result.data()
                    neo4j_ids.append(records[0]['neo4j_id'] if records else None)
                except:
                    neo4j_ids.append(None)
            return neo4j_ids
    
    # ... [REST OF THE CODE REMAINS SAME - phase2, phase3, etc.] ...
    
    async def phase2_import_semantic_scholar(self, target_papers: int = 6000000):
        """Phase 2: Import papers from Semantic Scholar API."""
        print("="*80)
        print("PHASE 2: IMPORTING SEMANTIC SCHOLAR")
        print("="*80)
        print(f"Target: {target_papers:,} new papers\n")
        
        # Search topics (same as before)
        topics = [
            "machine learning", "deep learning", "neural networks",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "computer vision", "image recognition", "object detection",
            "natural language processing", "language models", "transformers",
            # ... (keep all your topics)
        ]
        
        added = 0
        skipped = 0
        
        print(f"Searching {len(topics)} topics...\n")
        
        for topic in tqdm(topics, desc="Topics"):
            if added >= target_papers:
                break
            
            for offset in range(0, 10000, 100):
                if added >= target_papers:
                    break
                
                papers = await self._search_s2(topic, limit=100, offset=offset)
                
                if not papers:
                    break
                
                for paper in papers:
                    if added >= target_papers:
                        break
                    
                    title = paper.get('title') or ''
                    abstract = paper.get('abstract') or ''
                    
                    title = title.strip() if title else ''
                    abstract = abstract.strip() if abstract else ''
                    
                    s2_id = paper.get('paperId', '')
                    
                    if not title or not abstract or len(abstract) < 100 or not s2_id:
                        continue
                    
                    # Check duplicate
                    norm_title = self.normalize_title(title)
                    doi = self.normalize_doi(paper.get('externalIds', {}).get('DOI', ''))
                    arxiv_id = paper.get('externalIds', {}).get('ArXiv', '')
                    
                    if (norm_title in self.title_to_id or 
                        (doi and doi in self.doi_to_id) or
                        (arxiv_id and arxiv_id in self.arxiv_to_id) or
                        s2_id in self.s2_to_id):
                        skipped += 1
                        continue
                    
                    # Add to database
                    neo4j_id = await self._add_s2_paper(paper)
                    
                    if neo4j_id:
                        added += 1
                        
                        # Update indexes
                        self.title_to_id[norm_title] = neo4j_id
                        if doi:
                            self.doi_to_id[doi] = neo4j_id
                        if arxiv_id:
                            self.arxiv_to_id[arxiv_id] = neo4j_id
                        self.s2_to_id[s2_id] = neo4j_id
                        
                        if added % 1000 == 0:
                            print(f"\r  Added: {added:,} S2 papers (skipped {skipped:,})", end='', flush=True)
        
        print(f"\n\n✓ Phase 2 complete:")
        print(f"  Added:   {added:,} new S2 papers")
        print(f"  Skipped: {skipped:,} duplicates")
        print(f"  Total papers: {len(self.title_to_id):,}\n")
        
        return added
    
    async def _add_s2_paper(self, paper: Dict) -> Optional[str]:
        """Add single S2 paper using MERGE."""
        title = paper.get('title') or ''
        abstract = paper.get('abstract') or ''
        
        title = title.strip() if title else ''
        abstract = abstract.strip() if abstract else ''
        
        s2_id = paper.get('paperId', '')
        year = paper.get('year')
        venue = paper.get('venue', {}).get('name', '') if isinstance(paper.get('venue'), dict) else paper.get('venue', '')
        
        if not title or not abstract or len(abstract) < 100:
            return None
        
        external_ids = paper.get('externalIds', {})
        doi = self.normalize_doi(external_ids.get('DOI', ''))
        arxiv_id = external_ids.get('ArXiv', '')
        
        authors = paper.get('authors', [])
        author_names = [a.get('name', '') for a in authors if a.get('name')]
        
        unified_id = f"s2_{s2_id}"
        
        async with self.driver.session(database=self.database_name) as session:
            query = """
            MERGE (p:Paper {unified_id: $unified_id})
            ON CREATE SET
                p.title = $title,
                p.abstract = $abstract,
                p.year = $year,
                p.doi = $doi,
                p.arxiv_id = $arxiv_id,
                p.s2_paper_id = $s2_id,
                p.publication_name = $venue,
                p.source = 's2api'
            WITH p
            UNWIND $authors as author_name
            MERGE (a:Author {name: author_name})
            MERGE (a)-[:WROTE]->(p)
            RETURN elementId(p) as neo4j_id
            """
            
            try:
                result = await session.run(
                    query,
                    unified_id=unified_id,
                    title=title,
                    abstract=abstract,
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    s2_id=s2_id,
                    venue=venue,
                    authors=author_names
                )
                
                record = await result.single()
                return record['neo4j_id'] if record else None
            except:
                return None
    
    async def _search_s2(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Search Semantic Scholar."""
        await self._rate_limit()
        
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": "paperId,title,abstract,year,authors,venue,externalIds",
            "year": "2010-"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except:
            pass
        
        return []
    
    async def _get_s2_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get full paper details including citations."""
        await self._rate_limit()
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {
            "fields": "paperId,references,citations,references.paperId,citations.paperId"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        
        return None
    
    async def _create_citation(self, citing_id: str, cited_id: str):
        """Create citation relationship."""
        async with self.driver.session(database=self.database_name) as session:
            query = """
            MATCH (citing:Paper), (cited:Paper)
            WHERE elementId(citing) = $1 AND elementId(cited) = $2
            MERGE (citing)-[:CITES]->(cited)
            """
            await session.run(query, [citing_id, cited_id])
    
    async def _rate_limit(self):
        """Rate limiting for S2 API."""
        import time
        
        async with self.semaphore:
            if self.window_start is None:
                self.window_start = time.time()
            
            self.request_count += 1
            
            elapsed = time.time() - self.window_start
            if self.request_count >= 4500 and elapsed < 300:
                wait = 300 - elapsed + 1
                await asyncio.sleep(wait)
                self.request_count = 0
                self.window_start = time.time()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arxiv-file', type=str, required=True)
    parser.add_argument('--s2-api-key', type=str, required=True)
    parser.add_argument('--database', type=str, default='researchdb_hybrid')
    parser.add_argument('--arxiv-papers', type=int, default=2000000)
    parser.add_argument('--s2-papers', type=int, default=6000000)
    parser.add_argument('--skip-arxiv', action='store_true',
                       help='Skip arXiv import (if already loaded)')
    parser.add_argument('--skip-s2', action='store_true',
                       help='Skip S2 import (if already loaded)')
    parser.add_argument('--citations-only', action='store_true',
                       help='Only build citation network')
    args = parser.parse_args()
    
    print("="*80)
    print("HYBRID DATABASE BUILDER")
    print("="*80)
    print(f"\narXiv file: {args.arxiv_file}")
    print(f"S2 API key: {args.s2_api_key[:10]}...")
    print(f"Database: {args.database}")
    print(f"\nTarget papers:")
    print(f"  arXiv: {args.arxiv_papers:,}")
    print(f"  S2:    {args.s2_papers:,}")
    print(f"  Total: {args.arxiv_papers + args.s2_papers:,}")
    print(f"\nEstimated time: ~5 hours")
    print("="*80 + "\n")
    
    input("Press Enter to start building...")
    
    async with HybridDatabaseBuilder(args.s2_api_key, args.database) as builder:
        await builder.setup_database()
        
        # ✅ Check existing data
        existing = await builder.check_existing_data()
        
        # ✅ Load existing papers into memory
        if existing['total'] > 0:
            await builder.load_existing_indexes()
        
        # ✅ Smart phase selection
        arxiv_count = existing['arxiv']
        s2_count = existing['s2']
        
        # Phase 1: arXiv (skip if already loaded or --skip-arxiv)
        if not args.skip_arxiv and not args.citations_only:
            if existing['arxiv'] >= args.arxiv_papers:
                print(f"✓ arXiv already populated ({existing['arxiv']:,} papers), skipping Phase 1\n")
            else:
                new_arxiv = await builder.phase1_import_arxiv(args.arxiv_file, args.arxiv_papers)
                arxiv_count = existing['arxiv'] + new_arxiv
        
        # Phase 2: S2 (skip if already loaded or --skip-s2)
        if not args.skip_s2 and not args.citations_only:
            if existing['s2'] >= args.s2_papers:
                print(f"✓ S2 already populated ({existing['s2']:,} papers), skipping Phase 2\n")
            else:
                target_new = args.s2_papers - existing['s2']
                new_s2 = await builder.phase2_import_semantic_scholar(target_new)
                s2_count = existing['s2'] + new_s2
        
        # Phase 3: Citations (always run unless sufficient)
        if existing['citations'] < 100000:
            citation_count = await builder.phase3_link_citations()
        else:
            print(f"✓ Citations already populated ({existing['citations']:,}), skipping Phase 3\n")
            citation_count = existing['citations']
        
        total = arxiv_count + s2_count
        
        print("="*80)
        print("BUILD COMPLETE!")
        print("="*80)
        print(f"\nDatabase: {args.database}")
        print(f"Total papers: {total:,}")
        print(f"  arXiv:  {arxiv_count:,} ({arxiv_count/total*100:.1f}%)")
        print(f"  S2:     {s2_count:,} ({s2_count/total*100:.1f}%)")
        print(f"Citations: {citation_count:,}")
        print(f"\nNext steps:")
        print(f"  1. Update store.py: database='{args.database}'")
        print(f"  2. Rebuild cache: python rebuild_training_cache.py")
        print(f"  3. Communities: python -m RL.community_detection")
        print(f"  4. Train: python -m RL.train_rl --episodes 100")


if __name__ == "__main__":
    asyncio.run(main())
