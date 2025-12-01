import asyncio
import pickle
import os
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from graph.database.store import EnhancedStore


class CommunityDetector:
    """
    Detects and caches communities in the paper graph using Louvain algorithm.
    Communities are precomputed and stored for fast lookup during training.
    """
    
    def __init__(self, store: EnhancedStore, cache_file: str = "communities.pkl"):
        self.store = store
        self.cache_file = cache_file
        self.communities = {}  # node_id -> community_id
        self.community_sizes = {}  # community_id -> size
        self.is_loaded = False
    
    async def build_communities(self, method: str = "cypher", max_nodes: int = 50000):
        """
        Build community structure using Neo4j's native algorithms or Python-based clustering.
        
        Args:
            method: "cypher" (fast, uses Neo4j GDS) or "networkx" (slower, more control)
            max_nodes: Maximum nodes to process (for memory efficiency)
        """
        print(f"Building communities using {method} method...")
        
        if method == "cypher":
            await self._build_with_cypher_gds(max_nodes)
        elif method == "networkx":
            await self._build_with_networkx(max_nodes)
        else:
            await self._build_simple_citation_based(max_nodes)
        
        # Save to cache
        self._save_cache()
        self.is_loaded = True
        
        print(f"✓ Found {len(set(self.communities.values()))} communities")
        print(f"✓ Covering {len(self.communities)} nodes")
    
    async def _build_with_cypher_gds(self, max_nodes: int):
        """
        Use Neo4j Graph Data Science library for fast community detection.
        Requires GDS plugin: https://neo4j.com/docs/graph-data-science/current/
        """
        print("Attempting to use Neo4j GDS (Graph Data Science) library...")
        
        # Check if GDS is available
        check_query = "CALL gds.list() YIELD name RETURN count(*) as count"
        try:
            result = await self.store._run_query_method(check_query, [])
            if not result or result[0].get('count', 0) == 0:
                print("⚠ GDS not available, falling back to simple method")
                await self._build_simple_citation_based(max_nodes)
                return
        except Exception as e:
            print(f"⚠ GDS check failed: {e}, using simple method")
            await self._build_simple_citation_based(max_nodes)
            return
        
        # Project graph (papers connected by citations)
        project_query = """
            CALL gds.graph.project(
                'paper-citation-graph',
                'Paper',
                'CITES',
                {nodeProperties: ['year']}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        """
        
        try:
            # Drop existing projection if exists
            drop_query = "CALL gds.graph.drop('paper-citation-graph', false)"
            await self.store._run_query_method(drop_query, [])
        except:
            pass
        
        try:
            result = await self.store._run_query_method(project_query, [])
            print(f"✓ Graph projected: {result[0]['nodeCount']} nodes, {result[0]['relationshipCount']} edges")
        except Exception as e:
            print(f"✗ Graph projection failed: {e}")
            await self._build_simple_citation_based(max_nodes)
            return
        
        # Run Louvain community detection
        louvain_query = """
            CALL gds.louvain.stream('paper-citation-graph')
            YIELD nodeId, communityId
            MATCH (p:Paper) WHERE elementId(p) = nodeId
            RETURN elementId(p) as node_id, communityId as community_id
            LIMIT $1
        """
        
        try:
            results = await self.store._run_query_method(louvain_query, [max_nodes])
            
            for row in results:
                node_id = row['node_id']
                comm_id = row['community_id']
                self.communities[node_id] = comm_id
            
            # Calculate community sizes
            self._calculate_community_sizes()
            
            # Cleanup
            drop_query = "CALL gds.graph.drop('paper-citation-graph')"
            await self.store._run_query_method(drop_query, [])
            
        except Exception as e:
            print(f"✗ Louvain failed: {e}")
            await self._build_simple_citation_based(max_nodes)
    
    async def _build_with_networkx(self, max_nodes: int):
        """
        Use NetworkX + python-louvain for community detection.
        More flexible but slower than Neo4j GDS.
        """
        try:
            import networkx as nx
            from community import community_louvain
        except ImportError:
            print("⚠ NetworkX or python-louvain not installed")
            print("  Install with: pip install networkx python-louvain")
            await self._build_simple_citation_based(max_nodes)
            return
        
        print("Building graph with NetworkX...")
        
        # Fetch edges (citation relationships)
        query = """
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            RETURN elementId(p1) as source, elementId(p2) as target
            LIMIT $1
        """
        
        edges = await self.store._run_query_method(query, [max_nodes * 2])
        
        # Build NetworkX graph
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge['source'], edge['target'])
        
        print(f"✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Run Louvain
        print("Running Louvain algorithm...")
        partition = community_louvain.best_partition(G, random_state=42)
        
        self.communities = partition
        self._calculate_community_sizes()
    
    async def _build_simple_citation_based(self, max_nodes: int):
        """
        Simple fallback: cluster by citation patterns without external libraries.
        Uses connected components and citation counts.
        
        IMPORTANT: Uses elementId(p) to match the format used during navigation!
        """
        print("Using simple citation-based clustering...")
        
        # Get papers with their citation counts
        # CRITICAL: Use elementId(p) as node_id to match navigation format
        query = """
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
        
        papers = await self.store._run_query_method(query, [max_nodes])
        
        print(f"✓ Fetched {len(papers)} papers from database")
        
        if papers:
            # Show sample to verify format
            sample = papers[0]
            print(f"  Sample node_id format: {sample.get('node_id')}")
            print(f"  Sample title: {sample.get('title', 'N/A')[:50]}")
        
        # Assign communities based on citation patterns
        # Strategy: Group papers with similar citation profiles
        for paper in papers:
            node_id = paper['node_id']
            refs = paper.get('refs', 0)
            cites = paper.get('cites', 0)
            year = paper.get('year', 2000)
            
            # Create community ID based on:
            # 1. Citation tier (low/medium/high cited)
            # 2. Time period (5-year buckets)
            
            # More granular citation tiers
            if cites >= 1000:
                cite_tier = 5  # Extremely highly cited
            elif cites >= 100:
                cite_tier = 4  # Highly cited
            elif cites >= 20:
                cite_tier = 3  # Medium-high cited
            elif cites >= 5:
                cite_tier = 2  # Medium cited
            else:
                cite_tier = 1  # Low cited
            
            # 5-year time buckets for finer granularity
            if year:
                year_bucket = (year // 5) * 5  # 2015, 2020, 2025, etc.
            else:
                year_bucket = 2000
            
            # Community ID: "{year_bucket}_{cite_tier}"
            # Example: "2015_4" = papers from 2015-2019 with 100-999 citations
            comm_id = f"{year_bucket}_{cite_tier}"
            
            self.communities[node_id] = comm_id
        
        self._calculate_community_sizes()
        
        num_communities = len(set(self.communities.values()))
        print(f"✓ Created {num_communities} citation-based communities")
        print(f"✓ Covered {len(self.communities)} papers")
        
        # Show community distribution
        if self.community_sizes:
            sorted_sizes = sorted(self.community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  Top 5 largest communities:")
            for i, (comm_id, size) in enumerate(sorted_sizes, 1):
                percentage = (size / len(self.communities)) * 100
                print(f"    {i}. {comm_id}: {size} papers ({percentage:.1f}%)")
    
    def _calculate_community_sizes(self):
        """Calculate the size of each community."""
        comm_counter = Counter(self.communities.values())
        self.community_sizes = dict(comm_counter)
    
    def _save_cache(self):
        """Save communities to disk for fast loading."""
        cache_data = {
            'communities': self.communities,
            'community_sizes': self.community_sizes
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✓ Communities cached to {self.cache_file}")
    
    def load_cache(self) -> bool:
        """Load precomputed communities from disk."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.communities = cache_data['communities']
            self.community_sizes = cache_data['community_sizes']
            self.is_loaded = True
            
            print(f"✓ Loaded {len(set(self.communities.values()))} communities from cache")
            return True
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            return False
    
    def get_community(self, node_id: str) -> Optional[str]:
        """Get community ID for a node."""
        return self.communities.get(node_id, None)
    
    def get_community_size(self, community_id: str) -> int:
        """Get size of a community."""
        return self.community_sizes.get(community_id, 0)
    
    def get_statistics(self) -> Dict:
        """Get community statistics."""
        num_communities = len(set(self.communities.values()))
        avg_size = np.mean(list(self.community_sizes.values())) if self.community_sizes else 0
        max_size = max(self.community_sizes.values()) if self.community_sizes else 0
        min_size = min(self.community_sizes.values()) if self.community_sizes else 0
        
        return {
            'num_communities': num_communities,
            'num_nodes': len(self.communities),
            'avg_community_size': avg_size,
            'max_community_size': max_size,
            'min_community_size': min_size,
            'coverage': len(self.communities)  # How many nodes have communities
        }
    
    async def get_community_neighbors(self, community_id: str, limit: int = 10) -> List[str]:
        """
        Get neighboring communities (communities connected to this one).
        Useful for understanding community structure.
        """
        # Get nodes in this community
        nodes_in_comm = [nid for nid, cid in self.communities.items() if cid == community_id]
        
        if not nodes_in_comm:
            return []
        
        # Sample some nodes to check their neighbors
        sample_nodes = nodes_in_comm[:min(10, len(nodes_in_comm))]
        
        neighbor_communities = set()
        
        for node_id in sample_nodes:
            # Get neighbors of this node
            query = """
                MATCH (p:Paper)-[:CITES|CITED_BY]-(neighbor:Paper)
                WHERE elementId(p) = $1
                RETURN DISTINCT elementId(neighbor) as neighbor_id
                LIMIT 20
            """
            
            neighbors = await self.store._run_query_method(query, [node_id])
            
            for n in neighbors:
                n_id = n['neighbor_id']
                n_comm = self.get_community(n_id)
                if n_comm and n_comm != community_id:
                    neighbor_communities.add(n_comm)
        
        return list(neighbor_communities)[:limit]


async def build_and_cache_communities():
    """
    Standalone script to precompute communities.
    Run this once before training.
    """
    print("=" * 80)
    print("COMMUNITY DETECTION - Building Graph Communities")
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
        print("This may take 5-10 minutes for large graphs...")
        
        # Build communities (try GDS first, fallback to simple)
        await detector.build_communities(method="simple", max_nodes=50000)
        
        stats = detector.get_statistics()
        print(f"\n✓ Community detection complete!")
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Show example communities
    print(f"\n" + "=" * 80)
    print("SAMPLE COMMUNITIES")
    print("=" * 80)
    
    communities = defaultdict(list)
    for node_id, comm_id in list(detector.communities.items())[:100]:
        communities[comm_id].append(node_id)
    
    for i, (comm_id, nodes) in enumerate(list(communities.items())[:5], 1):
        print(f"\nCommunity {comm_id}:")
        print(f"  Size: {detector.get_community_size(comm_id)}")
        print(f"  Sample nodes: {len(nodes)}")
    
    await store.pool.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(build_and_cache_communities())