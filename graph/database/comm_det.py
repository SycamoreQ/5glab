import asyncio
import pickle
import os
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from graph.database.store import EnhancedStore


class CommunityDetector:
    """
    Detects communities for BOTH papers and authors.
    
    Paper Communities: Clustered by citation patterns + time period
    Author Communities: Clustered by collaboration patterns + research area
    """
    
    def __init__(self, store: EnhancedStore, cache_file: str = "communities_unified.pkl"):
        self.store = store
        self.cache_file = cache_file
        
        # Separate tracking for papers and authors
        self.paper_communities = {}  # paper_id -> community_id
        self.author_communities = {}  # author_id -> community_id
        
        self.paper_community_sizes = {}
        self.author_community_sizes = {}
        
        self.is_loaded = False
    
    async def build_communities(self, max_papers: int = None, max_authors: int = 20000):
        """
        Build communities for both papers and authors.
        
        Args:
            max_papers: Maximum papers to process (None = ALL papers in DB)
            max_authors: Maximum authors to process
        """
        print("Building unified communities (Papers + Authors)")
        
        await self._build_paper_communities(max_papers)
        
        await self._build_author_communities(max_authors)

        self._save_cache()
        self.is_loaded = True
        
        print("COMMUNITY DETECTION COMPLETE")
        print(f"Paper communities: {len(set(self.paper_communities.values()))}")
        print(f"Author communities: {len(set(self.author_communities.values()))}")
        print(f"Total papers covered: {len(self.paper_communities)}")
        print(f"Total authors covered: {len(self.author_communities)}")

    async def _build_paper_communities(self, max_papers: Optional[int]):
        """
        Build paper communities for TRAINING papers specifically.
        """
        print("\nPAPER COMMUNITIES")
        
        training_file = "training_papers.pkl"
        if os.path.exists(training_file):
            print(f"✓ Loading training papers from {training_file}...")
            with open(training_file, 'rb') as f:
                training_papers = pickle.load(f)
            
            paper_ids = [p['paper_id'] for p in training_papers]
            print(f"  Found {len(paper_ids)} training papers")
            print(f"  Sample IDs: {paper_ids[:3]}")
            
            print(f"\n  Fetching metadata for training papers...")
            
            query_training = """
                MATCH (p:Paper)
                WHERE elementId(p) IN $1
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                RETURN elementId(p) as node_id,
                    count(DISTINCT ref) as refs,
                    count(DISTINCT citing) as cites,
                    p.year as year,
                    COALESCE(p.title, p.id, '') as title
            """
            
            all_papers = await self.store._run_query_method(query_training, [paper_ids])
            print(f"  ✓ Fetched {len(all_papers)} papers")
            
        else:
            print(f" No training_papers.pkl found, falling back to batch fetch...")
            batch_size = 50000  
            skip = 0
            all_papers = []
            
            while True:
                print(f" Fetching batch {skip//batch_size + 1}...")
                
                query_batch = """
                    MATCH (p:Paper)
                    WITH p, elementId(p) as node_id
                    ORDER BY node_id
                    SKIP $skip
                    LIMIT $limit
                    OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                    WITH node_id,
                        count(DISTINCT ref) as ref_count,
                        count(DISTINCT citing) as cite_count,
                        p.year as year,
                        COALESCE(p.title, p.id, '') as title
                    RETURN node_id, ref_count as refs, cite_count as cites, year, title
                """
                
                batch = await self.store._run_query_method(query_batch)
                
                if not batch:
                    break
                
                all_papers.extend(batch)
                print(f"  Fetched {len(batch):,} papers (total: {len(all_papers):,})")
                
                skip += batch_size
                
                if max_papers and len(all_papers) >= max_papers:
                    all_papers = all_papers[:max_papers]
                    break
                
                await asyncio.sleep(0.1)
        
        if not all_papers:
            print("  No papers found!")
            return
        
        # Show sample
        if all_papers:
            top_cited = max(all_papers, key=lambda p: p.get('cites', 0))
            print(f"\n  Sample paper (highest citations):")
            print(f"    Title: {top_cited.get('title', 'Unknown')[:60]}")
            print(f"    ID: {top_cited['node_id']}")
            print(f"    Citations: {top_cited.get('cites', 0)}, References: {top_cited.get('refs', 0)}")
        
        print(f"\n  Assigning {len(all_papers):,} papers to communities...")
        
        for i, paper in enumerate(all_papers):
            node_id = paper['node_id']
            cites = paper.get('cites', 0)
            year = paper.get('year', 2000)
            
            if cites >= 1000:
                cite_tier = 5
            elif cites >= 100:
                cite_tier = 4
            elif cites >= 20:
                cite_tier = 3
            elif cites >= 5:
                cite_tier = 2
            else:
                cite_tier = 1
            
            year_bucket = (year // 5) * 5 if year else 2000
            
            comm_id = f"P_{year_bucket}_{cite_tier}"
            self.paper_communities[node_id] = comm_id
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{len(all_papers)} papers...")
        
        paper_counter = Counter(self.paper_communities.values())
        self.paper_community_sizes = dict(paper_counter)
        
        print(f"\n  ✓ Created {len(set(self.paper_communities.values()))} paper communities")
        print(f"  ✓ Covered {len(self.paper_communities):,} papers")
        
        top_paper_comms = sorted(self.paper_community_sizes.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 paper communities:")
        for i, (comm_id, size) in enumerate(top_paper_comms, 1):
            parts = comm_id.split('_')
            year = parts[1] if len(parts) > 1 else "?"
            tier = parts[2] if len(parts) > 2 else "?"
            tier_label = {
                '5': '1000+ cites',
                '4': '100-999 cites',
                '3': '20-99 cites',
                '2': '5-19 cites',
                '1': '<5 cites'
            }.get(tier, tier)
            print(f"    {i}. {comm_id}: {size:,} papers (Year ~{year}, {tier_label})")


    
    async def _build_author_communities(self, max_authors: int):
        """Build author communities based on collaboration patterns."""
        print("\nAUTHOR COMMUNITIES")
        print("-" * 80)
        print("  Note: Computing collaboration via co-authorship (shared papers)")
        
        all_authors = []
        
        print("  Phase 1/3: Fetching prolific authors (10+ papers)...")
        query_prolific = """
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            WITH a, elementId(a) as node_id, count(p) as paper_count
            WHERE paper_count >= 10
            OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
            WHERE a <> collab
            WITH node_id, paper_count, count(DISTINCT collab) as collab_count,
                 a.name as name, a.affiliation as affiliation
            RETURN node_id, paper_count, collab_count, name, affiliation
            LIMIT 5000
        """
        
        try:
            prolific = await self.store._run_query_method(query_prolific, [])
            all_authors.extend(prolific)
            print(f" Found {len(prolific)} prolific authors")
        except Exception as e:
            print(f" Phase 1 failed: {e}")
        
        if len(all_authors) < max_authors:
            print("  Phase 2/3: Fetching active authors (5-9 papers)...")
            remaining = max_authors - len(all_authors)
            
            query_active = """
                MATCH (a:Author)-[:WROTE]->(p:Paper)
                WITH a, elementId(a) as node_id, count(p) as paper_count
                WHERE paper_count >= 5 AND paper_count < 10
                OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
                WHERE a <> collab
                WITH node_id, paper_count, count(DISTINCT collab) as collab_count,
                     a.name as name, a.affiliation as affiliation
                RETURN node_id, paper_count, collab_count, name, affiliation
                LIMIT $1
            """
            
            try:
                active = await self.store._run_query_method(query_active, [remaining])
                all_authors.extend(active)
                print(f"Found {len(active)} active authors")
            except Exception as e:
                print(f"Phase 2 failed: {e}")
        
        if len(all_authors) < max_authors:
            print("  Phase 3/3: Fetching remaining authors...")
            remaining = max_authors - len(all_authors)
            
            query_rest = """
                MATCH (a:Author)-[:WROTE]->(p:Paper)
                WITH a, elementId(a) as node_id, count(p) as paper_count
                OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
                WHERE a <> collab
                WITH node_id, paper_count, count(DISTINCT collab) as collab_count,
                     a.name as name, a.affiliation as affiliation
                RETURN node_id, paper_count, collab_count, name, affiliation
                LIMIT $1
            """
            
            try:
                rest = await self.store._run_query_method(query_rest, [remaining])
                all_authors.extend(rest)
                print(f"Found {len(rest)} additional authors")
            except Exception as e:
                print(f"Phase 3 failed: {e}")
        
        print(f"\n  ✓ Total authors fetched: {len(all_authors):,}")
        
        if all_authors:
            sample = all_authors[0]
            print(f"  Sample author: {sample.get('name', 'Unknown')[:40]}")
            print(f"    Papers: {sample.get('paper_count', 0)}")
            print(f"    Collaborators: {sample.get('collab_count', 0)}")
        
        # Assign author communities
        for author in all_authors:
            node_id = author['node_id']
            paper_count = author.get('paper_count', 0)
            collab_count = author.get('collab_count', 0)
            
            if paper_count >= 50:
                prod_tier = 5
            elif paper_count >= 20:
                prod_tier = 4
            elif paper_count >= 10:
                prod_tier = 3
            elif paper_count >= 5:
                prod_tier = 2
            else:
                prod_tier = 1
            
            if collab_count >= 50:
                collab_tier = 3
            elif collab_count >= 10:
                collab_tier = 2
            else:
                collab_tier = 1
            
            comm_id = f"A_{prod_tier}_{collab_tier}"
            self.author_communities[node_id] = comm_id
        
        author_counter = Counter(self.author_communities.values())
        self.author_community_sizes = dict(author_counter)
        
        print(f"\n Created {len(set(self.author_communities.values()))} author communities")
        print(f"Covered {len(self.author_communities):,} authors")
        
        # Show distribution
        top_author_comms = sorted(self.author_community_sizes.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 author communities:")
        for i, (comm_id, size) in enumerate(top_author_comms, 1):
            parts = comm_id.split('_')
            if len(parts) == 3:
                prod = parts[1]
                collab = parts[2]
                prod_label = {
                    '5': 'Very Prolific (50+)',
                    '4': 'Prolific (20-49)',
                    '3': 'Active (10-19)',
                    '2': 'Moderate (5-9)',
                    '1': 'Occasional (<5)'
                }.get(prod, prod)
                collab_label = {
                    '3': 'Highly Collab (50+)',
                    '2': 'Collaborative (10-49)',
                    '1': 'Solo/Small (<10)'
                }.get(collab, collab)
                print(f"    {i}. {comm_id}: {size:,} authors ({prod_label}, {collab_label})")
    
    def _save_cache(self):
        """Save unified cache."""
        cache_data = {
            'paper_communities': self.paper_communities,
            'author_communities': self.author_communities,
            'paper_community_sizes': self.paper_community_sizes,
            'author_community_sizes': self.author_community_sizes
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"\n✓ Unified communities cached to {self.cache_file}")
    
    def load_cache(self) -> bool:
        """Load unified cache."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.paper_communities = cache_data['paper_communities']
            self.author_communities = cache_data['author_communities']
            self.paper_community_sizes = cache_data['paper_community_sizes']
            self.author_community_sizes = cache_data['author_community_sizes']
            self.is_loaded = True
            
            print(f"✓ Loaded unified communities from cache")
            print(f"  Papers: {len(self.paper_communities):,} in {len(set(self.paper_communities.values()))} communities")
            print(f"  Authors: {len(self.author_communities):,} in {len(set(self.author_communities.values()))} communities")
            return True
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            return False
    
    def get_community(self, node_id: str, node_type: str = None) -> Optional[str]:
        """Get community for a node."""
        if node_type == "paper" or node_id in self.paper_communities:
            return self.paper_communities.get(node_id)
        elif node_type == "author" or node_id in self.author_communities:
            return self.author_communities.get(node_id)
        else:
            return self.paper_communities.get(node_id) or self.author_communities.get(node_id)
    
    def get_community_size(self, community_id: str) -> int:
        """Get size of a community."""
        if community_id.startswith('P_'):
            return self.paper_community_sizes.get(community_id, 0)
        elif community_id.startswith('A_'):
            return self.author_community_sizes.get(community_id, 0)
        else:
            return 0
    
    def get_statistics(self) -> Dict:
        """Get unified statistics."""
        return {
            'num_paper_communities': len(set(self.paper_communities.values())),
            'num_author_communities': len(set(self.author_communities.values())),
            'num_papers': len(self.paper_communities),
            'num_authors': len(self.author_communities),
            'avg_paper_community_size': np.mean(list(self.paper_community_sizes.values())) if self.paper_community_sizes else 0,
            'avg_author_community_size': np.mean(list(self.author_community_sizes.values())) if self.author_community_sizes else 0,
        }


async def build_and_cache_unified_communities():
    """Build unified communities for papers and authors."""
    print("Unified Community Detection")
    print("Building communities for ALL Papers and Authors in database")
    
    store = EnhancedStore()
    detector = CommunityDetector(store)
    
    if detector.load_cache():
        print("\n✓ Communities already cached!")
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠ No cache found, building communities...")
        print("This may take 2-5 minutes depending on database size...")
        
        await detector.build_communities(max_papers=None, max_authors=20000)
        
        stats = detector.get_statistics()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    await store.pool.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(build_and_cache_unified_communities())
