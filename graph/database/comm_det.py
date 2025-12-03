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
    
    async def build_communities(self, max_papers: int = 50000, max_authors: int = 20000):
        """Build communities for both papers and authors."""
        print("=" * 80)
        print("BUILDING UNIFIED COMMUNITIES (Papers + Authors)")
        print("=" * 80)
        
        # Build paper communities
        await self._build_paper_communities(max_papers)
        
        # Build author communities
        await self._build_author_communities(max_authors)
        
        # Save to cache
        self._save_cache()
        self.is_loaded = True
        
        print("\n" + "=" * 80)
        print("COMMUNITY DETECTION COMPLETE")
        print("=" * 80)
        print(f"✓ Paper communities: {len(set(self.paper_communities.values()))}")
        print(f"✓ Author communities: {len(set(self.author_communities.values()))}")
        print(f"✓ Total papers covered: {len(self.paper_communities)}")
        print(f"✓ Total authors covered: {len(self.author_communities)}")
    
    async def _build_paper_communities(self, max_papers: int):
        """Build paper communities based on citation patterns."""
        print("\nPAPER COMMUNITIES")
        print("-" * 80)
        
        all_papers = []
        
        # Phase 1: Highly cited papers
        print("  Phase 1/3: Fetching highly-cited papers (100+ citations)...")
        query_highly_cited = """
            MATCH (p:Paper)<-[r:CITES]-(citing:Paper)
            WITH p, elementId(p) as node_id, count(r) as cite_count
            WHERE cite_count >= 100
            OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
            WITH node_id, cite_count, count(ref) as ref_count, p.year as year,
                 COALESCE(p.title, p.id, '') as title
            RETURN node_id, ref_count as refs, cite_count as cites, year, title
            LIMIT 5000
        """
        
        try:
            highly_cited = await self.store._run_query_method(query_highly_cited, [])
            all_papers.extend(highly_cited)
            print(f"    ✓ Found {len(highly_cited)} highly-cited papers")
        except Exception as e:
            print(f"    ⚠ Phase 1 failed: {e}")
        
        # Phase 2: Medium cited papers
        if len(all_papers) < max_papers:
            print("  Phase 2/3: Fetching medium-cited papers (5-99 citations)...")
            remaining = max_papers - len(all_papers)
            
            query_medium = """
                MATCH (p:Paper)<-[r:CITES]-(citing:Paper)
                WITH p, elementId(p) as node_id, count(r) as cite_count
                WHERE cite_count >= 5 AND cite_count < 100
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                WITH node_id, cite_count, count(ref) as ref_count, p.year as year,
                     COALESCE(p.title, p.id, '') as title
                RETURN node_id, ref_count as refs, cite_count as cites, year, title
                LIMIT $1
            """
            
            try:
                medium = await self.store._run_query_method(query_medium, [remaining])
                all_papers.extend(medium)
                print(f"Found {len(medium)} medium-cited papers")
            except Exception as e:
                print(f"Phase 2 failed: {e}")
        
        # Phase 3: Rest
        if len(all_papers) < max_papers:
            print("  Phase 3/3: Fetching remaining papers...")
            remaining = max_papers - len(all_papers)
            
            query_rest = """
                MATCH (p:Paper)
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                WITH p, elementId(p) as node_id, 
                     count(DISTINCT ref) as refs, 
                     count(DISTINCT citing) as cites
                WHERE refs > 0 OR cites > 0
                RETURN node_id, refs, cites, p.year as year,
                       COALESCE(p.title, p.id, '') as title
                LIMIT $1
            """
            
            try:
                rest = await self.store._run_query_method(query_rest, [remaining])
                all_papers.extend(rest)
                print(f"    ✓ Found {len(rest)} additional papers")
            except Exception as e:
                print(f"    ⚠ Phase 3 failed: {e}")
        
        # Assign paper communities
        for paper in all_papers:
            node_id = paper['node_id']
            cites = paper.get('cites', 0)
            year = paper.get('year', 2000)
            
            # Citation tier
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
        
        paper_counter = Counter(self.paper_communities.values())
        self.paper_community_sizes = dict(paper_counter)
        
        print(f"\n  ✓ Created {len(set(self.paper_communities.values()))} paper communities")
        print(f"  ✓ Covered {len(self.paper_communities)} papers")
        
        # Show distribution
        top_paper_comms = sorted(self.paper_community_sizes.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  Top 3 paper communities:")
        for i, (comm_id, size) in enumerate(top_paper_comms, 1):
            print(f"    {i}. {comm_id}: {size} papers")
    
    async def _build_author_communities(self, max_authors: int):
        """
        Build author communities based on collaboration patterns.
        
        IMPORTANT: Authors have NO direct edges. Collaboration is computed via:
        (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        """
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
            print(f"    ✓ Found {len(prolific)} prolific authors")
        except Exception as e:
            print(f"    ⚠ Phase 1 failed: {e}")
        
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
                print(f"    ✓ Found {len(active)} active authors")
            except Exception as e:
                print(f"    ⚠ Phase 2 failed: {e}")
        
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
                print(f"    ✓ Found {len(rest)} additional authors")
            except Exception as e:
                print(f"    ⚠ Phase 3 failed: {e}")
        
        print(f"\n  ✓ Total authors fetched: {len(all_authors)}")
        
        if all_authors:
            sample = all_authors[0]
            print(f"  Sample author: {sample.get('name', 'Unknown')[:40]}")
            print(f"    Papers: {sample.get('paper_count', 0)}")
            print(f"    Collaborators (via shared papers): {sample.get('collab_count', 0)}")
        
        
        for author in all_authors:
            node_id = author['node_id']
            paper_count = author.get('paper_count', 0)
            collab_count = author.get('collab_count', 0)
            
            # Productivity tier
            if paper_count >= 50:
                prod_tier = 5  # Very prolific
            elif paper_count >= 20:
                prod_tier = 4  # Prolific
            elif paper_count >= 10:
                prod_tier = 3  # Active
            elif paper_count >= 5:
                prod_tier = 2  # Moderate
            else:
                prod_tier = 1  # Occasional
            
            # Collaboration tier (via co-authorship)
            if collab_count >= 50:
                collab_tier = 3  # Highly collaborative
            elif collab_count >= 10:
                collab_tier = 2  # Collaborative
            else:
                collab_tier = 1  # Solo/small team
            
            # Community ID: "A_{productivity}_{collaboration}"
            # Examples:
            # - A_5_3 = Very prolific (50+ papers), highly collaborative (50+ co-authors)
            # - A_2_1 = Moderate (5-9 papers), solo researcher
            comm_id = f"A_{prod_tier}_{collab_tier}"
            
            self.author_communities[node_id] = comm_id
        
        # Calculate sizes
        author_counter = Counter(self.author_communities.values())
        self.author_community_sizes = dict(author_counter)
        
        print(f"\n  ✓ Created {len(set(self.author_communities.values()))} author communities")
        print(f"  ✓ Covered {len(self.author_communities)} authors")
        
        # Show distribution
        top_author_comms = sorted(self.author_community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 author communities:")
        for i, (comm_id, size) in enumerate(top_author_comms, 1):
            # Decode community ID
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
                print(f"    {i}. {comm_id}: {size} authors ({prod_label}, {collab_label})")
            else:
                print(f"    {i}. {comm_id}: {size} authors")
    
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
            print(f"  Papers: {len(self.paper_communities)} in {len(set(self.paper_communities.values()))} communities")
            print(f"  Authors: {len(self.author_communities)} in {len(set(self.author_communities.values()))} communities")
            return True
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            return False
    
    def get_community(self, node_id: str, node_type: str = None) -> Optional[str]:
        """
        Get community for a node.
        Auto-detects type from paper/author communities if node_type not provided.
        """
        if node_type == "paper" or node_id in self.paper_communities:
            return self.paper_communities.get(node_id)
        elif node_type == "author" or node_id in self.author_communities:
            return self.author_communities.get(node_id)
        else:
            # Try both
            return self.paper_communities.get(node_id) or self.author_communities.get(node_id)
    
    def get_community_size(self, community_id: str) -> int:
        """Get size of a community (works for both paper and author communities)."""
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
    print("=" * 80)
    print("UNIFIED COMMUNITY DETECTION")
    print("Building communities for Papers AND Authors")
    print("=" * 80)
    
    store = EnhancedStore()
    detector = CommunityDetector(store)
    
    # Try to load existing cache
    if detector.load_cache():
        print("\n✓ Communities already cached!")
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠ No cache found, building communities...")
        print("This may take 10-15 minutes for large graphs...")
        
        await detector.build_communities(max_papers=50000, max_authors=20000)
        
        stats = detector.get_statistics()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    await store.pool.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(build_and_cache_unified_communities())