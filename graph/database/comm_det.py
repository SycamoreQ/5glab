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
        else:
            await self._build_simple_citation_based(max_nodes)
        
        # Save to cache
        self._save_cache()
        self.is_loaded = True
        
        print(f"✓ Found {len(set(self.communities.values()))} communities")
        print(f"✓ Covering {len(self.communities)} nodes")
    
    async def _build_with_cypher_gds(self, max_nodes: int):
        print("Attempting to use Neo4j GDS Louvain")
        
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
        

        project_query = f"""
                CALL gds.graph.project.cypher(
                    'paper-citation-graph',
                    'MATCH (p:Paper) WHERE COUNT {{ (p)<-[:CITES]-() }} > 5 RETURN id(p) AS id',
                    'MATCH (p1:Paper)-[:CITES]->(p2:Paper) 
                    WHERE COUNT {{ (p1)<-[:CITES]-() }} > 5 
                    AND COUNT {{ (p2)<-[:CITES]-() }} > 5 
                    RETURN id(p1) AS source, id(p2) AS target'
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """

        
        try:
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
        
        louvain_query = """
            CALL gds.leiden.stream('paper-citation-graph', {
                maxLevels: 10,
                tolerance: 0.0001,
                gamma: 1.0
            })
            YIELD nodeId, communityId
            MATCH (p:Paper) WHERE id(p) = nodeId
            RETURN elementId(p) as node_id, 
                communityId as community_id,
                COUNT { (p)<-[:CITES]-() } as cites,
                p.year as year
            LIMIT $1
            """
        
        try:
            results = await self.store._run_query_method(louvain_query, [max_nodes])
            
            for row in results:
                node_id = row['node_id']
                comm_id = row['community_id']
                self.communities[node_id] = comm_id
            
            self._calculate_community_sizes()
            
            drop_query = "CALL gds.graph.drop('paper-citation-graph')"
            await self.store._run_query_method(drop_query, [])
            
        except Exception as e:
            print(f"✗ Louvain failed: {e}")
            await self._build_simple_citation_based(max_nodes)




    async def _build_simple_citation_based(self, max_nodes: int):

        print("Building connected ego-network communities...")
        
        query = """
        MATCH (center:Paper)
        WHERE center.year IS NOT NULL 
        AND COUNT { (center)<-[:CITES]-() } > 10
        WITH center
        ORDER BY COUNT { (center)<-[:CITES]-() } DESC
        LIMIT 500
        
        // Get center + its 1-hop neighborhood
        CALL {
            WITH center
            MATCH (center)-[:CITES|CITED_BY]-(neighbor:Paper)
            RETURN DISTINCT elementId(neighbor) as node_id,
                COUNT { (neighbor)<-[:CITES]-() } as cites,
                neighbor.year as year
            LIMIT 100
            
            UNION
            
            WITH center
            RETURN elementId(center) as node_id,
                COUNT { (center)<-[:CITES]-() } as cites,
                center.year as year
        }
        
        RETURN DISTINCT node_id, cites, year
        LIMIT $1
        """
        
        papers = await self.store._run_query_method(query, [max_nodes])
        
        if not papers:
            print("✗ No papers found")
            return
        
        print(f"✓ Fetched {len(papers):,} papers from database")
        
        if papers:
            sample = papers[0]
            print(f"  Sample node_id format: {sample.get('node_id')}")
        
        # Assign communities
        for paper in papers:
            node_id = paper['node_id']
            cites = paper.get('cites', 0)
            year = paper.get('year', 2000)
            
            # Assign community based on citation patterns
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
            comm_id = f"{year_bucket}_{cite_tier}"
            
            self.communities[node_id] = comm_id
        
        self._calculate_community_sizes()
        num_communities = len(set(self.communities.values()))
        print(f"✓ Created {num_communities} citation-based communities")
        print(f"✓ Covered {len(self.communities)} papers")
        
        if self.community_sizes:
            sorted_sizes = sorted(self.community_sizes.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
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
            'coverage': len(self.communities) 
        }
    
    async def get_community_neighbors(self, community_id: str, limit: int = 10) -> List[str]:
        """
        Get neighboring communities (communities connected to this one).
        Useful for understanding community structure.
        """
        nodes_in_comm = [nid for nid, cid in self.communities.items() if cid == community_id]
        
        if not nodes_in_comm:
            return []
        
        sample_nodes = nodes_in_comm[:min(10, len(nodes_in_comm))]
        
        neighbor_communities = set()
        
        for node_id in sample_nodes:
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
        
        await detector.build_communities(method="simple", max_nodes=1000000)
        
        stats = detector.get_statistics()
        print(f"\n✓ Community detection complete!")
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
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