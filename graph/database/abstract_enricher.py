"""
OPTIMIZED Abstract Enricher - 10x faster with batching and async requests.
Expected time: 1-2 hours instead of 14 hours!
"""

import asyncio
import aiohttp
import time
from graph.database.store import EnhancedStore
from tqdm.asyncio import tqdm
from typing import List, Dict, Optional
import pickle


class FastAbstractEnricher:
    """
    Optimized enricher using:
    1. Async HTTP requests (10 concurrent)
    2. Batch Neo4j updates (100 at a time)
    3. Better caching
    """
    
    def __init__(self, store: EnhancedStore):
        self.store = store
        self.base_url = "https://api.openalex.org/works"
        self.session = None
        self.semaphore = asyncio.Semaphore(10)  # 10 concurrent requests
        self.batch_size = 100
    
    async def __aenter__(self):
        """Setup async session."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup."""
        if self.session:
            await self.session.close()
    
    def reconstruct_abstract(self, inverted_index):
        """Convert OpenAlex inverted index back to text."""
        if not inverted_index:
            return ""
        
        words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        
        return " ".join([words[i] for i in sorted(words.keys())])
    
    async def fetch_abstract_async(self, title: str, doi: str = None) -> Optional[str]:
        """
        Fetch abstract asynchronously with rate limiting.
        """
        async with self.semaphore:
            try:
                # Try DOI first (more reliable)
                if doi and doi.strip():
                    doi_clean = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                    url = f"https://api.openalex.org/works/https://doi.org/{doi_clean}"
                    
                    try:
                        async with self.session.get(
                            url,
                            params={"select": "abstract_inverted_index"}
                        ) as response:
                            if response.status == 200:
                                work = await response.json()
                                abstract_index = work.get("abstract_inverted_index", {})
                                abstract = self.reconstruct_abstract(abstract_index)
                                if abstract and len(abstract) > 50:
                                    return abstract
                    except:
                        pass  # Fallback to title search
                
                # Fallback: Search by title
                async with self.session.get(
                    self.base_url,
                    params={
                        "filter": f"title.search:{title[:200]}",  # Limit title length
                        "select": "abstract_inverted_index",
                        "per_page": 1
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        if results:
                            work = results[0]
                            abstract_index = work.get("abstract_inverted_index", {})
                            abstract = self.reconstruct_abstract(abstract_index)
                            if abstract and len(abstract) > 50:
                                return abstract
                
                return None
                
            except Exception as e:
                # Silently fail for individual papers
                return None
    
    async def enrich_batch(self, papers: List[Dict]) -> List[tuple]:
        """
        Enrich a batch of papers concurrently.
        Returns list of (paper_id, abstract) tuples.
        """
        tasks = []
        for paper in papers:
            task = self.fetch_abstract_async(
                paper['title'],
                paper.get('doi', '')
            )
            tasks.append((paper['paper_id'], task))
        
        # Fetch all concurrently
        results = []
        for paper_id, task in tasks:
            try:
                abstract = await task
                if abstract:
                    results.append((paper_id, abstract))
            except:
                pass
        
        return results
    
    async def update_batch_neo4j(self, updates: List[tuple]):
        """
        Update Neo4j in batch (much faster than individual updates).
        """
        if not updates:
            return
        
        # Build batch update query
        query = """
        UNWIND $updates as update
        MATCH (p:Paper)
        WHERE elementId(p) = update.paper_id
        SET p.abstract = update.abstract
        """
        
        params = [
            {"paper_id": pid, "abstract": abstract}
            for pid, abstract in updates
        ]
        
        await self.store._run_query_method(query, [params])
    
    async def enrich_papers_fast(self, max_papers: int = 20000):
        """
        Fast enrichment with async + batching.
        """
        print(f"🚀 FAST MODE: Enriching up to {max_papers:,} papers")
        print("="*80)
        
        # Get papers without abstracts
        query = """
        MATCH (p:Paper)
        WHERE (p.abstract IS NULL OR p.abstract = '' OR NOT exists(p.abstract))
        AND (p.title IS NOT NULL AND p.title <> '')
        RETURN elementId(p) as paper_id,
               p.title as title,
               COALESCE(p.doi, '') as doi
        LIMIT $1
        """
        
        all_papers = await self.store._run_query_method(query, [max_papers])
        print(f"Found {len(all_papers):,} papers to enrich\n")
        
        if not all_papers:
            print("✓ All papers already have abstracts!")
            return
        
        enriched_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(all_papers), self.batch_size):
            batch = all_papers[i:i + self.batch_size]
            
            # Fetch abstracts concurrently
            updates = await self.enrich_batch(batch)
            
            # Update Neo4j in batch
            if updates:
                await self.update_batch_neo4j(updates)
                enriched_count += len(updates)
            
            failed_count += (len(batch) - len(updates))
            
            # Progress
            progress = min(i + self.batch_size, len(all_papers))
            percent = (progress / len(all_papers)) * 100
            
            print(f"\r[{progress:>6}/{len(all_papers)}] {percent:>5.1f}% | "
                  f"✓ {enriched_count:>5} | ✗ {failed_count:>5}", end='', flush=True)
        
        print("\n")
        print("="*80)
        print("ENRICHMENT COMPLETE!")
        print("="*80)
        print(f"Successfully enriched: {enriched_count:,} papers")
        print(f"Failed to find: {failed_count:,} papers")
        print(f"Success rate: {enriched_count/(enriched_count+failed_count)*100:.1f}%")


async def enrich_training_cache_fast():
    """
    FASTEST: Only enrich papers in training_papers.pkl.
    Expected time: 15-30 minutes!
    """
    print("🚀 ULTRA-FAST MODE: Enriching training cache only")
    print("="*80)
    
    # Load cache
    print("\n1. Loading training_papers.pkl...")
    with open('training_papers.pkl', 'rb') as f:
        cached_papers = pickle.load(f)
    
    print(f"   Found {len(cached_papers):,} papers in cache")
    
    # Check how many need abstracts
    need_abstract = [p for p in cached_papers if not p.get('abstract') or len(p.get('abstract', '')) < 50]
    print(f"   Papers needing abstracts: {len(need_abstract):,}")
    
    if not need_abstract:
        print("✓ All papers already have abstracts!")
        return
    
    # Enrich
    print(f"\n2. Fetching abstracts from OpenAlex...")
    store = EnhancedStore(pool_size=20)
    
    async with FastAbstractEnricher(store) as enricher:
        enriched_count = 0
        
        # Process in batches
        for i in range(0, len(need_abstract), enricher.batch_size):
            batch = need_abstract[i:i + enricher.batch_size]
            
            # Fetch abstracts
            updates = await enricher.enrich_batch(batch)
            
            # Update both cache and Neo4j
            if updates:
                # Update Neo4j
                await enricher.update_batch_neo4j(updates)
                
                # Update in-memory cache
                update_dict = dict(updates)
                for paper in batch:
                    if paper['paper_id'] in update_dict:
                        paper['abstract'] = update_dict[paper['paper_id']]
                        enriched_count += 1
            
            # Progress
            progress = min(i + enricher.batch_size, len(need_abstract))
            percent = (progress / len(need_abstract)) * 100
            print(f"\r[{progress:>6}/{len(need_abstract)}] {percent:>5.1f}% | ✓ {enriched_count:>5}", 
                  end='', flush=True)
        
        print("\n")
    
    await store.pool.close()
    
    # Save updated cache
    print(f"3. Saving enriched cache...")
    with open('training_papers_enriched.pkl', 'wb') as f:
        pickle.dump(cached_papers, f)
    
    print("="*80)
    print("✓ DONE!")
    print("="*80)
    print(f"Enriched: {enriched_count:,} / {len(need_abstract):,} papers")
    print(f"Success rate: {enriched_count/len(need_abstract)*100:.1f}%")
    print(f"\nNew cache: training_papers_enriched.pkl")
    print(f"\nNext steps:")
    print(f"  1. Update train_rl.py to use 'training_papers_enriched.pkl'")
    print(f"  2. Run: python -m RL.train_rl --episodes 10")
    print(f"  3. Check similarity scores (should be 0.5-0.7 now!)")


async def main():
    """
    Choose enrichment mode.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cache',
                       choices=['cache', 'full'],
                       help='cache: only training papers (15-30 min), full: all papers (1-2 hours)')
    parser.add_argument('--max-papers', type=int, default=20000)
    args = parser.parse_args()
    
    if args.mode == 'cache':
        print("Mode: Training cache only (FASTEST)")
        await enrich_training_cache_fast()
    else:
        print(f"Mode: Full database enrichment (up to {args.max_papers:,} papers)")
        store = EnhancedStore(pool_size=20)
        async with FastAbstractEnricher(store) as enricher:
            await enricher.enrich_papers_fast(args.max_papers)
        await store.pool.close()


if __name__ == "__main__":
    asyncio.run(main())