"""
Add CITES relationships to existing papers using Semantic Scholar API.

This script:
1. Fetches citation data from S2 API using arXiv IDs
2. Creates CITES relationships between papers in your database
3. Works with your existing 2M arXiv papers

Timeline: ~4-6 hours to add citations for 2M papers
"""

import asyncio
import aiohttp
from typing import Dict, List, Set, Optional
from tqdm.asyncio import tqdm
from neo4j import AsyncGraphDatabase


class CitationLinker:
    """
    Add citations to existing papers using S2 API.
    """
    
    def __init__(self, s2_api_key: str, database_name: str):
        self.s2_api_key = s2_api_key
        self.database_name = database_name
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Indexes for fast lookup
        self.arxiv_to_neo4j = {}  # arxiv_id -> neo4j_id
        self.s2_to_neo4j = {}     # s2_paper_id -> neo4j_id
        self.doi_to_neo4j = {}    # doi -> neo4j_id
        self.title_to_neo4j = {}  # normalized_title -> neo4j_id
        
        self.driver = None
        self.session = None
        
        # Rate limiting (with API key: 5000 requests per 5 minutes)
        self.semaphore = asyncio.Semaphore(100)  # Increased from 80
        self.request_count = 0
        self.window_start = None
        
        # Batch citation creation for speed
        self.citation_batch = []
        self.citation_batch_size = 500  # Create citations in batches
    
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
        if not title:
            return ""
        title = title.lower().strip()
        title = ''.join(c for c in title if c.isalnum() or c.isspace())
        return ' '.join(title.split())
    
    async def build_paper_index(self):
        """
        Index all papers for fast lookup.
        """
        print("="*80)
        print("BUILDING PAPER INDEX")
        print("="*80)
        print(f"Database: {self.database_name}\n")
        
        print("Indexing papers for fast citation lookup...")
        
        async with self.driver.session(database=self.database_name) as session:
            # Fetch all papers with their identifiers
            query = """
            MATCH (p:Paper)
            RETURN elementId(p) as neo4j_id,
                   COALESCE(p.arxiv_id, '') as arxiv_id,
                   COALESCE(p.s2_paper_id, '') as s2_id,
                   COALESCE(p.doi, '') as doi,
                   p.title as title
            """
            
            result = await session.run(query)
            papers = await result.data()
            
            print(f"Indexing {len(papers):,} papers...\n")
            
            for paper in tqdm(papers, desc="Building index"):
                neo4j_id = paper['neo4j_id']
                
                # Index by arXiv ID
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id:
                    self.arxiv_to_neo4j[arxiv_id] = neo4j_id
                
                # Index by S2 ID
                s2_id = paper.get('s2_id', '')
                if s2_id:
                    self.s2_to_neo4j[s2_id] = neo4j_id
                
                # Index by DOI
                doi = paper.get('doi', '')
                if doi:
                    self.doi_to_neo4j[doi.lower().strip()] = neo4j_id
                
                # Index by title
                title = paper.get('title', '')
                if title:
                    norm_title = self.normalize_title(title)
                    self.title_to_neo4j[norm_title] = neo4j_id
        
        print(f"\n✓ Index built:")
        print(f"  arXiv IDs:  {len(self.arxiv_to_neo4j):,}")
        print(f"  S2 IDs:     {len(self.s2_to_neo4j):,}")
        print(f"  DOIs:       {len(self.doi_to_neo4j):,}")
        print(f"  Titles:     {len(self.title_to_neo4j):,}\n")
    
    async def add_citations(self, batch_size: int = 5000, max_papers: int = None):
        """
        Fetch citations from S2 API and create relationships.
        
        Args:
            batch_size: Papers to process in each batch (increased to 5000)
            max_papers: Limit papers to process (None = all)
        """
        print("="*80)
        print("ADDING CITATIONS")
        print("="*80)
        print("Fetching citation data from Semantic Scholar API\n")
        
        # Get papers with arXiv IDs (these are in S2)
        papers_to_process = [
            (arxiv_id, neo4j_id) 
            for arxiv_id, neo4j_id in list(self.arxiv_to_neo4j.items())[:max_papers]
        ]
        
        if not papers_to_process:
            print("⚠ No papers with arXiv IDs found!")
            return 0
        
        print(f"Processing {len(papers_to_process):,} papers")
        print(f"Batch size: {batch_size:,} (larger = faster)")
        print(f"Concurrent requests: 100\n")
        
        citation_count = 0
        papers_processed = 0
        papers_with_citations = 0
        
        for i in range(0, len(papers_to_process), batch_size):
            batch = papers_to_process[i:i+batch_size]
            
            # Process batch concurrently
            tasks = []
            for arxiv_id, citing_neo4j_id in batch:
                task = self._process_paper_citations(arxiv_id, citing_neo4j_id)
                tasks.append(task)
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            for result in results:
                if isinstance(result, tuple):
                    papers_processed += 1
                    added, has_cites = result
                    citation_count += added
                    if has_cites:
                        papers_with_citations += 1
            
            # Flush citation batch
            if self.citation_batch:
                await self._flush_citation_batch()
            
            print(f"\r  Papers: {papers_processed:,} | "
                  f"Citations: {citation_count:,} | "
                  f"Papers w/ cites: {papers_with_citations:,}", 
                  end='', flush=True)
        
        # Final flush
        if self.citation_batch:
            await self._flush_citation_batch()
        
        print(f"\n\n✓ Citation linking complete!")
        print(f"  Papers processed: {papers_processed:,}")
        print(f"  Papers with citations: {papers_with_citations:,}")
        print(f"  Citations added: {citation_count:,}")
        if papers_with_citations > 0:
            print(f"  Avg citations/paper: {citation_count/papers_with_citations:.1f}\n")
        
        return citation_count
    
    def _find_paper_in_db(self, paper_ref: Dict) -> Optional[str]:
        """
        Find a paper in our database using various identifiers.
        """
        if not paper_ref:
            return None
        
        # Try S2 ID
        s2_id = paper_ref.get('paperId', '')
        if s2_id and s2_id in self.s2_to_neo4j:
            return self.s2_to_neo4j[s2_id]
        
        # Try arXiv ID
        external_ids = paper_ref.get('externalIds') or {}
        arxiv_id = external_ids.get('ArXiv', '') if external_ids else ''
        if arxiv_id and arxiv_id in self.arxiv_to_neo4j:
            return self.arxiv_to_neo4j[arxiv_id]
        
        # Try DOI
        doi = external_ids.get('DOI', '') if external_ids else ''
        if doi:
            doi_normalized = doi.lower().strip()
            if doi_normalized in self.doi_to_neo4j:
                return self.doi_to_neo4j[doi_normalized]
        
        # Try title match
        title = paper_ref.get('title', '')
        if title:
            norm_title = self.normalize_title(title)
            if norm_title in self.title_to_neo4j:
                return self.title_to_neo4j[norm_title]
        
        return None
    
    async def _get_paper_by_arxiv(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper data from S2 using arXiv ID."""
        await self._rate_limit()
        
        url = f"{self.base_url}/paper/arXiv:{arxiv_id}"
        params = {
            "fields": "paperId,title,references,references.paperId,references.title,references.externalIds"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        
        return None
    
    async def _update_s2_id(self, neo4j_id: str, s2_id: str):
        """Update paper with S2 ID."""
        async with self.driver.session(database=self.database_name) as session:
            query = """
            MATCH (p:Paper)
            WHERE elementId(p) = $neo4j_id
            SET p.s2_paper_id = $s2_id
            """
            await session.run(query, {"neo4j_id": neo4j_id, "s2_id": s2_id})
    
    async def _create_citation(self, citing_id: str, cited_id: str):
        """Create CITES relationship."""
        async with self.driver.session(database=self.database_name) as session:
            query = """
            MATCH (citing:Paper), (cited:Paper)
            WHERE elementId(citing) = $citing_id AND elementId(cited) = $cited_id
            MERGE (citing)-[:CITES]->(cited)
            """
            await session.run(query, {"citing_id": citing_id, "cited_id": cited_id})
    
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
                print(f"\n⏳ Rate limit: waiting {wait:.0f}s...")
                await asyncio.sleep(wait)
                self.request_count = 0
                self.window_start = time.time()
    
    async def verify_citations(self):
        """Check citation network statistics."""
        print("="*80)
        print("CITATION NETWORK VERIFICATION")
        print("="*80)
        
        async with self.driver.session(database=self.database_name) as session:
            # Count total citations
            query_total = """
            MATCH ()-[r:CITES]->()
            RETURN count(r) as total_citations
            """
            result = await session.run(query_total)
            record = await result.single()
            total_cites = record['total_citations'] if record else 0
            
            # Count papers with citations
            query_with_cites = """
            MATCH (p:Paper)-[:CITES]->()
            RETURN count(DISTINCT p) as papers_with_refs
            """
            result = await session.run(query_with_cites)
            record = await result.single()
            with_refs = record['papers_with_refs'] if record else 0
            
            # Count cited papers
            query_cited = """
            MATCH ()-[:CITES]->(p:Paper)
            RETURN count(DISTINCT p) as cited_papers
            """
            result = await session.run(query_cited)
            record = await result.single()
            cited = record['cited_papers'] if record else 0
            
            # Total papers
            query_total_papers = """
            MATCH (p:Paper)
            RETURN count(p) as total
            """
            result = await session.run(query_total_papers)
            record = await result.single()
            total_papers = record['total'] if record else 0
            
            print(f"\nCitation Network Statistics:")
            print(f"  Total papers:           {total_papers:,}")
            print(f"  Papers citing others:   {with_refs:,} ({with_refs/total_papers*100:.1f}%)")
            print(f"  Papers being cited:     {cited:,} ({cited/total_papers*100:.1f}%)")
            print(f"  Total citations:        {total_cites:,}")
            print(f"  Avg citations/paper:    {total_cites/total_papers:.1f}")
            
            if total_cites == 0:
                print("\n⚠ WARNING: No citations found!")
                print("  Run the citation linker to add citations")
            elif with_refs / total_papers < 0.3:
                print("\n⚠ Low citation coverage (<30%)")
                print("  Consider running citation linker on more papers")
            else:
                print("\n✓ Citation network looks good!")


async def main():
    """
    Add citations to existing database.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2-api-key', type=str, required=True,
                       help='Semantic Scholar API key')
    parser.add_argument('--database', type=str, required=True,
                       help='Your database name (e.g., researchdb_hybrid)')
    parser.add_argument('--max-papers', type=int, default=None,
                       help='Max papers to process (None = all)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('--verify-only', action='store_true',
                       help='Just verify existing citations')
    args = parser.parse_args()
    
    print("="*80)
    print("CITATION LINKER")
    print("="*80)
    print(f"\nDatabase: {args.database}")
    print(f"S2 API key: {args.s2_api_key[:10]}...")
    
    if args.max_papers:
        print(f"Max papers: {args.max_papers:,}")
        print(f"Estimated time: ~{args.max_papers//1000} hours")
    else:
        print(f"Processing: ALL papers")
        print(f"Estimated time: ~4-6 hours for 2M papers")
    
    print("="*80 + "\n")
    
    async with CitationLinker(args.s2_api_key, args.database) as linker:
        if args.verify_only:
            await linker.build_paper_index()
            await linker.verify_citations()
        else:
            input("Press Enter to start adding citations...")
            
            await linker.build_paper_index()
            citation_count = await linker.add_citations(
                batch_size=args.batch_size,
                max_papers=args.max_papers
            )
            await linker.verify_citations()
    
    print("\n✓ Done!")
    print("\nNext steps:")
    print("  1. Rebuild cache: python rebuild_training_cache.py")
    print("  2. Communities: python -m RL.community_detection")
    print("  3. Train: python -m RL.train_rl --episodes 100")


if __name__ == "__main__":
    asyncio.run(main())