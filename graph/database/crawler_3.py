import asyncio
import aiohttp
import json
import hashlib
from typing import Dict, List, Set, Optional
from tqdm.asyncio import tqdm
from neo4j import AsyncGraphDatabase


class HybridDatabaseBuilder:
    """
    Build hybrid database combining arXiv + Semantic Scholar.
    """
    
    def __init__(self, s2_api_key: str, database_name: str = "researchdbv3"):
        self.s2_api_key = s2_api_key
        self.database_name = database_name
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Deduplication index
        self.title_to_id = {}  # normalized_title -> neo4j_id
        self.doi_to_id = {}    # doi -> neo4j_id
        self.arxiv_to_id = {}  # arxiv_id -> neo4j_id
        self.s2_to_id = {}     # s2_paper_id -> neo4j_id
        
        # Citation tracking (for Phase 3)
        self.pending_citations = []  # List of (citing_id, cited_id) tuples
        
        # Neo4j
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
        async with self.driver.session(database="researchdbv3") as session:
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
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.unified_id IS UNIQUE",
            ]
            for idx in indexes:
                try:
                    await session.run(idx)
                except:
                    pass  # Constraint might fail, that's ok
        
        print("✓ Database ready!\n")
    
    async def phase1_import_arxiv(self, arxiv_file: str, max_papers: int = 2000000):
        """
        Phase 1: Import arXiv papers.
        """
        print("="*80)
        print("PHASE 1: IMPORTING ARXIV")
        print("="*80)
        print(f"Target: {max_papers:,} papers\n")
        
        added = 0
        
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
                    
                    # Check duplicate
                    norm_title = self.normalize_title(title)
                    if norm_title in self.title_to_id:
                        continue
                    
                    # Prepare
                    arxiv_id = paper.get('id', '')
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
                    
                    added += 1
                    
                    # Batch insert
                    if len(batch) >= batch_size:
                        neo4j_ids = await self._insert_batch(session, batch)
                        
                        # Update dedup indexes
                        for paper_data, neo4j_id in zip(batch, neo4j_ids):
                            norm_title = self.normalize_title(paper_data['title'])
                            self.title_to_id[norm_title] = neo4j_id
                            
                            if paper_data['doi']:
                                self.doi_to_id[paper_data['doi']] = neo4j_id
                            if paper_data['arxiv_id']:
                                self.arxiv_to_id[paper_data['arxiv_id']] = neo4j_id
                        
                        batch = []
                
                # Insert remaining
                if batch:
                    neo4j_ids = await self._insert_batch(session, batch)
                    for paper_data, neo4j_id in zip(batch, neo4j_ids):
                        norm_title = self.normalize_title(paper_data['title'])
                        self.title_to_id[norm_title] = neo4j_id
                        if paper_data['doi']:
                            self.doi_to_id[paper_data['doi']] = neo4j_id
                        if paper_data['arxiv_id']:
                            self.arxiv_to_id[paper_data['arxiv_id']] = neo4j_id
        
        print(f"\n✓ Phase 1 complete: {added:,} arXiv papers imported")
        print(f"  Dedup index size: {len(self.title_to_id):,} titles\n")
        
        return added
    
    async def phase2_import_semantic_scholar(self, target_papers: int = 6000000):
        """
        Phase 2: Import papers from Semantic Scholar API.
        """
        print("="*80)
        print("PHASE 2: IMPORTING SEMANTIC SCHOLAR")
        print("="*80)
        print(f"Target: {target_papers:,} papers\n")
        
        # Search topics (expanded for 6M papers)
        topics = [
            # Core ML/AI
            "machine learning", "deep learning", "neural networks",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "transfer learning", "meta learning", "few shot learning",
            
            # Computer Vision
            "computer vision", "image recognition", "object detection",
            "image segmentation", "semantic segmentation", "instance segmentation",
            "video analysis", "action recognition", "pose estimation",
            "image generation", "style transfer", "super resolution",
            
            # NLP
            "natural language processing", "language models", "transformers",
            "text generation", "machine translation", "question answering",
            "sentiment analysis", "named entity recognition", "text classification",
            "dialogue systems", "chatbots", "language understanding",
            
            # Specific Architectures
            "convolutional neural networks", "recurrent neural networks",
            "attention mechanism", "graph neural networks",
            "generative adversarial networks", "variational autoencoders",
            "diffusion models", "vision transformers",
            
            # AI Applications
            "medical image analysis", "drug discovery", "bioinformatics",
            "autonomous driving", "robotics", "robot learning",
            "recommendation systems", "information retrieval",
            "speech recognition", "audio processing",
            
            # Other CS fields (for diversity)
            "data mining", "knowledge graphs", "information extraction",
            "distributed systems", "database systems", "software engineering",
            "human computer interaction", "computer graphics",
            "computational biology", "scientific computing",
        ]
        
        added = 0
        
        print(f"Searching {len(topics)} topics...\n")
        
        for topic in tqdm(topics, desc="Topics"):
            if added >= target_papers:
                break
            
            # Search with pagination
            for offset in range(0, 10000, 100):  # Up to 10K per topic
                if added >= target_papers:
                    break
                
                papers = await self._search_s2(topic, limit=100, offset=offset)
                
                if not papers:
                    break
                
                for paper in papers:
                    if added >= target_papers:
                        break
                    
                    # Extract
                    title = paper.get('title', '').strip()
                    abstract = paper.get('abstract', '').strip()
                    s2_id = paper.get('paperId', '')
                    
                    if not title or not abstract or len(abstract) < 100 or not s2_id:
                        continue
                    
                    # Check duplicate
                    norm_title = self.normalize_title(title)
                    doi = self.normalize_doi(paper.get('externalIds', {}).get('DOI', ''))
                    arxiv_id = paper.get('externalIds', {}).get('ArXiv', '')
                    
                    # Skip if already exists
                    if (norm_title in self.title_to_id or 
                        (doi and doi in self.doi_to_id) or
                        (arxiv_id and arxiv_id in self.arxiv_to_id) or
                        s2_id in self.s2_to_id):
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
                            print(f"\r  Added: {added:,} S2 papers", end='', flush=True)
        
        print(f"\n\n✓ Phase 2 complete: {added:,} S2 papers imported")
        print(f"  Total papers: {len(self.title_to_id):,}\n")
        
        return added
    
    async def phase3_link_citations(self):
        """
        Phase 3: Build citation network across both datasets.
        """
        print("="*80)
        print("PHASE 3: LINKING CITATIONS")
        print("="*80)
        print("Building unified citation network...\n")
        
        # Get all papers with their IDs
        async with self.driver.session(database=self.database_name) as session:
            query = """
            MATCH (p:Paper)
            WHERE p.s2_paper_id IS NOT NULL AND p.s2_paper_id <> ''
            RETURN elementId(p) as neo4j_id, p.s2_paper_id as s2_id
            LIMIT 100000
            """
            
            result = await session.run(query)
            papers = await result.data()
        
        print(f"Found {len(papers):,} papers with S2 IDs")
        print("Fetching citation data from S2 API...\n")
        
        citation_count = 0
        
        for i, paper in enumerate(tqdm(papers, desc="Processing citations")):
            neo4j_id = paper['neo4j_id']
            s2_id = paper['s2_id']
            
            # Get paper details with citations/references
            details = await self._get_s2_paper_details(s2_id)
            
            if not details:
                continue
            
            citing_id = neo4j_id
        
            references = details.get('references', [])
            for ref in references[:50]: 
                ref_s2_id = ref.get('paperId', '')
                
                if ref_s2_id in self.s2_to_id:
                    cited_id = self.s2_to_id[ref_s2_id]
                    await self._create_citation(citing_id, cited_id)
                    citation_count += 1
            
            # Process citations (other papers cite this)
            citations = details.get('citations', [])
            for cite in citations[:50]:  # Limit to 50 cites per paper
                cite_s2_id = cite.get('paperId', '')
                
                if cite_s2_id in self.s2_to_id:
                    citing_other_id = self.s2_to_id[cite_s2_id]
                    await self._create_citation(citing_other_id, citing_id)
                    citation_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"\r  Citations created: {citation_count:,}", end='', flush=True)
        
        print(f"\n\n✓ Phase 3 complete: {citation_count:,} citations linked\n")
        
        return citation_count
    
    async def _insert_batch(self, session, batch: List[Dict]) -> List[str]:
        """Insert batch and return Neo4j IDs."""
        query = """
        UNWIND $batch as paper
        CREATE (p:Paper {
            unified_id: paper.unified_id,
            title: paper.title,
            abstract: paper.abstract,
            year: paper.year,
            doi: paper.doi,
            arxiv_id: paper.arxiv_id,
            s2_paper_id: paper.s2_paper_id,
            keywords: paper.keywords,
            source: paper.source
        })
        WITH p, paper
        UNWIND paper.authors as author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:WROTE]->(p)
        RETURN elementId(p) as neo4j_id
        """
        
        result = await session.run(query, batch=batch)
        records = await result.data()
        return [r['neo4j_id'] for r in records]
    
    async def _add_s2_paper(self, paper: Dict) -> Optional[str]:
        """Add single S2 paper."""
        title = paper.get('title', '').strip()
        abstract = paper.get('abstract', '').strip()
        s2_id = paper.get('paperId', '')
        year = paper.get('year')
        venue = paper.get('venue', {}).get('name', '') if isinstance(paper.get('venue'), dict) else paper.get('venue', '')
        
        external_ids = paper.get('externalIds', {})
        doi = self.normalize_doi(external_ids.get('DOI', ''))
        arxiv_id = external_ids.get('ArXiv', '')
        
        authors = paper.get('authors', [])
        author_names = [a.get('name', '') for a in authors if a.get('name')]
        
        unified_id = f"s2_{s2_id}"
        
        async with self.driver.session(database=self.database_name) as session:
            query = """
            CREATE (p:Paper {
                unified_id: $1,
                title: $2,
                abstract: $3,
                year: $4,
                doi: $5,
                arxiv_id: $6,
                s2_paper_id: $7,
                publication_name: $8,
                source: 's2api'
            })
            WITH p
            UNWIND $9 as author_name
            MERGE (a:Author {name: author_name})
            MERGE (a)-[:WROTE]->(p)
            RETURN elementId(p) as neo4j_id
            """
            
            result = await session.run(
                query,
                [unified_id, title, abstract, year, doi, arxiv_id, s2_id, venue, author_names]
            )
            
            record = await result.single()
            return record['neo4j_id'] if record else None
    
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
    """
    Build hybrid database.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arxiv-file', type=str, required=True,
                       help='Path to arxiv-metadata-oai-snapshot.json')
    parser.add_argument('--s2-api-key', type=str, required=True,
                       help='Semantic Scholar API key')
    parser.add_argument('--database', type=str, default='researchdbv3',
                       help='Database name')
    parser.add_argument('--arxiv-papers', type=int, default=2000000,
                       help='Target arXiv papers')
    parser.add_argument('--s2-papers', type=int, default=6000000,
                       help='Target S2 papers')
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
        
        arxiv_count = await builder.phase1_import_arxiv(args.arxiv_file, args.arxiv_papers)
        s2_count = await builder.phase2_import_semantic_scholar(args.s2_papers)
        citation_count = await builder.phase3_link_citations()
        
        total = arxiv_count + s2_count
        
        print("="*80)
        print("BUILD COMPLETE!")
        print("="*80)
        print(f"\nDatabase: {args.database}")
        print(f"Total papers: {total:,}")
        print(f"  arXiv:  {arxiv_count:,} ({arxiv_count/total*100:.1f}%)")
        print(f"  S2:     {s2_count:,} ({s2_count/total*100:.1f}%)")
        print(f"Citations: {citation_count:,}")
        print(f"\nAbstract coverage: 100%")
        print(f"Expected similarity: 0.7-0.8")
        print(f"\nNext steps:")
        print(f"  1. Update store.py: database='{args.database}'")
        print(f"  2. Rebuild cache: python rebuild_training_cache.py")
        print(f"  3. Communities: python -m RL.community_detection")
        print(f"  4. Train: python -m RL.train_rl --episodes 100")


if __name__ == "__main__":
    asyncio.run(main())