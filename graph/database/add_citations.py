import asyncio
import aiohttp
import time
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
        self.semaphore = asyncio.Semaphore(15)  # Limits concurrency to 15 at a time
        self.request_count = 0
        self.window_start = None
        self.max_requests_per_window = 4800  # Conservative limit for 5000/5min
        self.window_duration = 300  # 5 minutes
        
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
    
    async def add_citations(self, batch_size: int = 1000, max_papers: int = None):
        """
        Fetch citations from S2 API and create relationships.
        
        Args:
            batch_size: Papers to process in each batch (default: 1000)
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
        
        total_papers = len(papers_to_process)
        print(f"Processing {total_papers:,} papers")
        print(f"Batch size: {batch_size:,}")
        print(f"Concurrent requests: 15 (rate-limited)")
        print(f"Estimated time: ~{total_papers//900} hours (Targeting ~15 req/s)\n")
        
        citation_count = 0
        papers_processed = 0
        papers_with_citations = 0

        # Use tqdm.asyncio for a clean, non-interrupting progress bar
        papers_iter = tqdm(
            desc="Processing papers",
            unit="paper",
            total=total_papers
        )
        
        for i in range(0, total_papers, batch_size):
            batch = papers_to_process[i:i+batch_size]
            
            # Process batch concurrently (but rate-limited)
            tasks = [
                self._process_paper_citations(arxiv_id, citing_neo4j_id)
                for arxiv_id, citing_neo4j_id in batch
            ]
            
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
            
            # Update TQDM progress
            papers_iter.update(len(batch))
            papers_iter.set_postfix(citations=f"{citation_count:,}", refresh=True)

        papers_iter.close()
        
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
    
    async def _process_paper_citations(self, arxiv_id: str, citing_neo4j_id: str) -> tuple:
        """Process citations for a single paper (async)."""
        try:
            # Get S2 data
            s2_data = await self._get_paper_by_arxiv(arxiv_id)
            
            if not s2_data:
                return (0, False)
            
            # Store S2 ID
            s2_id = s2_data.get('paperId', '')
            if s2_id:
                # Update local index (crucial for finding *cited* papers later)
                self.s2_to_neo4j[s2_id] = citing_neo4j_id
                # Update DB (can be done concurrently without awaiting)
                asyncio.create_task(self._update_s2_id(citing_neo4j_id, s2_id))
            
            # Process references
            references = s2_data.get('references', [])
            added = 0
            
            # Only process the first 100 references to save on processing time
            for ref in references[:100]:
                cited_neo4j_id = self._find_paper_in_db(ref)
                
                if cited_neo4j_id:
                    # Add to batch instead of creating immediately
                    self.citation_batch.append((citing_neo4j_id, cited_neo4j_id))
                    added += 1
            
            return (added, added > 0)
        
        except Exception as e:
            # print(f"Error processing paper {arxiv_id}: {e}") # Debugging
            return (0, False)
    
    async def _flush_citation_batch(self):
        """Create citations in batch for speed."""
        if not self.citation_batch:
            return
        
        async with self.driver.session(database=self.database_name) as session:
            query = """
            UNWIND $citations as citation
            MATCH (citing:Paper), (cited:Paper)
            WHERE elementId(citing) = citation.citing_id 
              AND elementId(cited) = citation.cited_id
            MERGE (citing)-[:CITES]->(cited)
            """
            
            citations_data = [
                {"citing_id": citing, "cited_id": cited}
                for citing, cited in self.citation_batch
            ]
            
            await session.run(query, {"citations": citations_data})
        
        self.citation_batch = []
    
    async def _get_paper_by_arxiv(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper data from S2 using arXiv ID."""
        await self._rate_limit()
        
        url = f"{self.base_url}/paper/arXiv:{arxiv_id}"
        # Request references with identifiers to maximize local matches
        params = {
            "fields": "paperId,references,references.paperId,references.title,references.externalIds"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Explicit 429 handler (should be caught by _rate_limit, but for safety)
                    print("\n⚠️ Received 429 from S2. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    # Simple retry (consider a more robust backoff strategy for production)
                    async with self.session.get(url, params=params) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.json()
        except aiohttp.ClientConnectorError:
             # Handle DNS/Connection issues gracefully
             pass 
        except Exception:
            # Catch other exceptions like JSONDecodeError, TimeoutError
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
    
    async def _rate_limit(self):
        """
        Proper rate limiting for S2 API using the 5000 requests/5 min rule.
        Uses a minimum inter-request delay for a smooth rate.
        """
        import time
        
        async with self.semaphore:
            current_time = time.time()
            
            # --- Window Reset Logic ---
            # If the current window has expired (5 minutes = 300 seconds)
            if self.window_start is None or (current_time - self.window_start) >= self.window_duration:
                # If we were in a cooldown, print a message
                if self.request_count >= self.max_requests_per_window:
                    # Print without leading \n to keep progress bar clean
                    print(f"✅ Rate limit window reset. Continuing...") 
                
                # Reset the window
                self.request_count = 0
                self.window_start = current_time
                current_time = self.window_start # Reset current_time for accurate calculations
            
            # --- Hard Limit Check and Wait ---
            if self.request_count >= self.max_requests_per_window:
                # Calculate remaining wait time until the window resets
                time_elapsed = current_time - self.window_start
                wait_time = self.window_duration - time_elapsed + 1 # +1 second buffer
                
                print(f"⏳ Rate limit reached ({self.request_count} requests in {time_elapsed:.0f}s). Waiting {wait_time:.0f}s for window reset...")
                await asyncio.sleep(wait_time)
                
                # After waiting, reset the window
                self.request_count = 0
                self.window_start = time.time()
            
            # --- Smooth Rate Delay (Throttle) ---
            # 4800 requests / 300 seconds = 1 request per 0.0625 seconds.
            # We enforce a *minimum* delay between requests to keep the rate smooth.
            min_delay = 0.07  # ~14.2 requests/sec (safe)
            await asyncio.sleep(min_delay)
            
            # Increment after the delay and checks
            self.request_count += 1
    
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
            
            if total_papers > 0:
                print(f"  Papers citing others:   {with_refs:,} ({with_refs/total_papers*100:.1f}%)")
                print(f"  Papers being cited:     {cited:,} ({cited/total_papers*100:.1f}%)")
                print(f"  Total citations:        {total_cites:,}")
                print(f"  Avg citations/paper:    {total_cites/total_papers:.1f}")
            else:
                print("  No papers found in the database.")

            
            if total_cites == 0 and total_papers > 0:
                print("\n⚠ WARNING: No citations found!")
                print("  Run the citation linker to add citations")
            elif total_papers > 0 and with_refs / total_papers < 0.3:
                print("\n⚠ Low citation coverage (<30%)")
                print("  Consider running citation linker on more papers")
            elif total_papers > 0:
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
        # Note: This ETA is an estimate based on a high-throughput scenario
        print(f"Estimated time: ~{args.max_papers//900} hours (Targeting ~15 req/s)") 
    else:
        print(f"Processing: ALL papers")
        print(f"Estimated time: ~4-6 hours for 2M papers (based on average citation lookup speed)")
    
    print("="*80 + "\n")
    
    async with CitationLinker(args.s2_api_key, args.database) as linker:
        if args.verify_only:
            await linker.build_paper_index()
            await linker.verify_citations()
        else:
            try:
                input("Press Enter to start adding citations...")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return
            
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")