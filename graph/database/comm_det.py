import asyncio
import pickle
import os
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from graph.database.store import EnhancedStore

class CommunityDetector:

    def __init__(self, store: EnhancedStore, cache_file: str = "communities_unified.pkl", use_leiden: bool = True):
        self.store = store
        self.cache_file = cache_file
        self.paper_communities = {}
        self.author_communities = {}
        self.paper_community_sizes = {}
        self.author_community_sizes = {}
        self.is_loaded = False
        self.use_leiden = use_leiden

    async def build_communities(self, max_papers: int = None, max_authors: int = 20000):
        print(f"Building unified communities (Papers + Authors)")
        print(f"Method: {'Leiden (Graph-based)' if self.use_leiden else 'Tier-based (Metadata)'}")
        
        if self.use_leiden:
            await self._build_paper_communities_leiden(max_papers)
        else:
            await self._build_paper_communities_tier_based(max_papers)
        
        await self._build_author_communities(max_authors)
        self._save_cache()
        self.is_loaded = True
        
        print("\nCOMMUNITY DETECTION COMPLETE")
        print(f"Paper communities: {len(set(self.paper_communities.values()))}")
        print(f"Author communities: {len(set(self.author_communities.values()))}")
        print(f"Total papers covered: {len(self.paper_communities)}")
        print(f"Total authors covered: {len(self.author_communities)}")


    async def _build_paper_communities_leiden(self, max_papers: Optional[int]):
        print("\nPAPER communities using Leiden")

        training_file = "training_papers.pkl"
        if not os.path.exists(training_file):
            print(f"{training_file} not found! Run cache_once.py first.")
            await self._build_paper_communities_tier_based(max_papers)
            return
        
        print(f" Loading cached paper IDs from {training_file}...")
        with open(training_file, 'rb') as f:
            cached_papers = pickle.load(f)
        
        paper_ids = [p['paper_id'] for p in cached_papers]
        print(f" Loaded {len(paper_ids):,} paper IDs from cache")
        
        print("\n  Dropping existing graph projection...")
        drop_query = "CALL gds.graph.drop('paperCitationGraph', false)"
        try:
            await self.store._run_query_method(drop_query, [])
            print(" Dropped existing projection")
        except:
            print("  No existing projection")

        print(f"\n  Creating graph projection (CACHED PAPERS ONLY)...")
        print(f"  Projecting {len(paper_ids):,} papers and their citations...")


        project_query = f"""
            CALL gds.graph.project.cypher(
            'paperCitationGraph',
            'MATCH (p:Paper) WHERE p.paperId IN $paper_ids RETURN id(p) AS id',
            'MATCH (p1:Paper)-[:CITES]->(p2:Paper) 
            WHERE p1.paperId IN $paper_ids AND p2.paperId IN $paper_ids 
            RETURN id(p1) AS source, id(p2) AS target
            UNION ALL
            MATCH (p1:Paper)-[:CITES]->(p2:Paper) 
            WHERE p1.paperId IN $paper_ids AND p2.paperId IN $paper_ids 
            RETURN id(p2) AS source, id(p1) AS target',
            {{parameters: {{paper_ids: $1}}}}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
        
        node_count = 0
        rel_count = 0
        
        try:
            result = await self.store._run_query_method(project_query, [paper_ids])
            if result:
                node_count = result[0]['nodeCount']
                rel_count = result[0]['relationshipCount']
                print(f" Graph projected: {node_count:,} nodes, {rel_count:,} edges (undirected)")
                
                if node_count != len(paper_ids):
                    print(f" WARNING: Expected {len(paper_ids):,} nodes, got {node_count:,}")
                
                if rel_count < 10000:
                    print(f" WARNING: Very sparse graph ({rel_count} edges). Falling back to tier-based.")
                    await self.store._run_query_method(drop_query, [])
                    await self._build_paper_communities_tier_based(max_papers)
                    return
        except Exception as e:
            print(f" ERROR: Graph projection failed: {e}")
            print(" Falling back to tier-based method...")
            await self._build_paper_communities_tier_based(max_papers)
            return
        
        print(f"\n  Running Leiden algorithm (may take 3-5 minutes for {node_count:,} nodes)...")
        leiden_query = """
        CALL gds.louvain.stream('paperCitationGraph', {
        maxLevels: 10,
        concurrency: 4
        })
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).paperId AS paperId, communityId
        """
        
        try:
            leiden_results = await self.store._run_query_method(leiden_query, [])
            print(f" Leiden completed: {len(leiden_results):,} node assignments")
            
            if not leiden_results:
                print(" WARNING: Leiden returned no results. Falling back to tier-based.")
                await self.store._run_query_method(drop_query, [])
                await self._build_paper_communities_tier_based(max_papers)
                return
        except Exception as e:
            print(f" ERROR: Leiden failed: {e}")
            await self.store._run_query_method(drop_query, [])
            await self._build_paper_communities_tier_based(max_papers)
            return

        print("\n  Assigning base communities...")
        for result in leiden_results:
            paper_id = result.get('paperId')
            comm_id = result.get('communityId')
            if paper_id and comm_id is not None:
                self.paper_communities[paper_id] = f"L_{comm_id}"
        
        base_communities = len(set(self.paper_communities.values()))
        print(f" Assigned {len(self.paper_communities):,} papers to {base_communities} base communities")
        
        print("\n  Enriching with metadata")
        batch_size = 5000
        metadata_map = {}
        
        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i:i+batch_size]
            query = """
            MATCH (p:Paper)
            WHERE p.paperId IN $1
            RETURN p.paperId as paper_id,
                p.year as year,
                COALESCE(p.citationCount, 0) as cites,
                p.fieldsOfStudy as fields
            """
            results = await self.store._run_query_method(query, [batch])
            for r in results:
                pid = r['paper_id']
                metadata_map[pid] = r
            
            if (i + batch_size) % 20000 == 0:
                print(f"  Fetched metadata for {len(metadata_map):,}/{len(paper_ids):,} papers...")
        
        print(f" Metadata fetched for {len(metadata_map):,} papers")
        
        for paper_id, comm_id in list(self.paper_communities.items()):
            meta = metadata_map.get(paper_id, {})
            year = meta.get('year', 'UNK')
            cites = meta.get('cites', 0) or 0
            fields = meta.get('fields', []) or []
            
            field = fields[0][:3].upper() if isinstance(fields, list) and fields else "GEN"
            cite_tier = 'H' if cites >= 100 else 'M' if cites >= 20 else 'L'
            
            enriched_id = f"{comm_id}_{year}_{cite_tier}_{field}"
            self.paper_communities[paper_id] = enriched_id
        
        print("\n  Cleaning up graph projection...")
        try:
            await self.store._run_query_method(drop_query, [])
            print("  Cleanup complete")
        except:
            pass
        
        paper_counter = Counter(self.paper_communities.values())
        self.paper_community_sizes = dict(paper_counter)
        
        enriched_communities = len(set(self.paper_communities.values()))
        print(f"\n✓ Created {enriched_communities:,} enriched paper communities (from {base_communities} base communities)")
        print(f" Covered {len(self.paper_communities):,} papers")
        
        top_paper_comms = sorted(self.paper_community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n Top 10 paper communities:")
        for i, (comm_id, size) in enumerate(top_paper_comms, 1):
            print(f"  {i}. {comm_id}: {size:,} papers")




    async def _build_paper_communities_tier_based(self, max_papers: Optional[int]):
        """Fallback: Tier-based community detection using metadata."""
        print("\nPAPER COMMUNITIES (Tier-Based)")
        print("-" * 80)
        
        training_file = "training_papers.pkl"
        all_papers = []
        
        if os.path.exists(training_file):
            print(f"  ✓ Loading from cache: {training_file}")
            try:
                with open(training_file, 'rb') as f:
                    cached_papers = pickle.load(f)
                
                if max_papers:
                    cached_papers = cached_papers[:max_papers]
                
                print(f"Found {len(cached_papers):,} cached papers")
                print(f"Fetching metadata from Neo4j...")
                
                batch_size = 5000
                for i in range(0, len(cached_papers), batch_size):
                    batch_papers = cached_papers[i:i+batch_size]
                    paper_ids = [p['paper_id'] for p in batch_papers]
                    
                    query = """
                    MATCH (p:Paper)
                    WHERE p.paperId IN $1
                    RETURN p.paperId as node_id,
                           COALESCE(p.citationCount, 0) as cites,
                           p.year as year,
                           p.fieldsOfStudy as fields
                    """
                    batch_result = await self.store._run_query_method(query, [paper_ids])
                    all_papers.extend(batch_result)
                    
                    if (i + batch_size) % 20000 == 0:
                        print(f"      Fetched {len(all_papers):,}/{len(cached_papers):,} papers...")
                    await asyncio.sleep(0.1)
                
                print(f"Fetched {len(all_papers):,} papers from Neo4j")
                
            except Exception as e:
                print(f"    Cache load failed: {e}")
                all_papers = []
        
        if not all_papers:
            print(f"Scanning database in batches...")
            batch_size = 10000
            skip = 0
            
            while True:
                print(f"    Fetching batch at offset {skip:,}...")
                query = """
                MATCH (p:Paper)
                WHERE p.title IS NOT NULL AND p.year IS NOT NULL
                WITH p ORDER BY p.paperId
                SKIP $1
                LIMIT $2
                RETURN p.paperId as node_id,
                       COALESCE(p.citationCount, 0) as cites,
                       p.year as year,
                       p.fieldsOfStudy as fields
                """
                try:
                    batch = await self.store._run_query_method(query, [skip, batch_size])
                except Exception as e:
                    print(f"    Batch failed: {e}")
                    break
                
                if not batch:
                    break
                
                all_papers.extend(batch)
                print(f"Total: {len(all_papers):,} papers")
                skip += batch_size
                
                if max_papers and len(all_papers) >= max_papers:
                    all_papers = all_papers[:max_papers]
                    break
                
                await asyncio.sleep(0.2)
        
        if not all_papers:
            print("    No papers found!")
            return
        
        top_cited = max(all_papers, key=lambda p: p.get('cites', 0) or 0)
        print(f"\n Sample paper (highest citations):")
        print(f" Title: {top_cited.get('title', 'Unknown')[:60] if 'title' in top_cited else 'N/A'}")
        print(f" ID: {top_cited.get('node_id', 'N/A')}")
        print(f" Citations: {top_cited.get('cites', 0)}")
        
        print(f"\n  Assigning {len(all_papers):,} papers to communities...")
        
        for i, paper in enumerate(all_papers):
            node_id = paper.get('node_id')
            if not node_id:
                continue
            
            cites = paper.get('cites', 0) or 0
            year = paper.get('year')
            fields = paper.get('fields', []) or []
            
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
            
            if year and isinstance(year, int) and 1950 <= year <= 2030:
                year_bucket = (year // 5) * 5
            else:
                year_bucket = "UNK"
            
            primary_field = "General"
            if isinstance(fields, list) and fields:
                primary_field = fields[0] if isinstance(fields[0], str) else "General"
            elif isinstance(fields, str):
                primary_field = fields
            
            field_short = primary_field[:3].upper() if primary_field != "General" else "GEN"
            comm_id = f"P_{year_bucket}_{cite_tier}_{field_short}"
            self.paper_communities[node_id] = comm_id
            
            if (i + 1) % 20000 == 0:
                print(f"    Progress: {i+1}/{len(all_papers)} papers...")
        
        paper_counter = Counter(self.paper_communities.values())
        self.paper_community_sizes = dict(paper_counter)
        
        print(f"\nCreated {len(set(self.paper_communities.values()))} paper communities")
        print(f"Covered {len(self.paper_communities):,} papers")
        
        top_paper_comms = sorted(self.paper_community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n Top 10 paper communities:")
        for i, (comm_id, size) in enumerate(top_paper_comms, 1):
            print(f"    {i}. {comm_id}: {size:,} papers")

    async def _build_author_communities(self, max_authors: int):
        """Build author communities using collaboration patterns."""
        print("\nAUTHOR COMMUNITIES")
        print("-" * 80)
        
        all_authors = []
        author_file = 'training_authors.pkl'
        
        if os.path.exists(author_file):
            print(f" Loading from cache: {author_file}")
            try:
                with open(author_file, 'rb') as f:
                    all_authors = pickle.load(f)
                if max_authors:
                    all_authors = all_authors[:max_authors]
                print(f"    Found {len(all_authors):,} cached authors")
            except Exception as e:
                print(f"    Cache load failed: {e}")
                all_authors = []
        
        if not all_authors:
            print("  Fetching from database...")
            query_prolific = """
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            WITH a, count(p) as paper_count
            WHERE paper_count >= 5
            OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
            WHERE a <> collab
            WITH a.authorId as node_id,
                 paper_count,
                 count(DISTINCT collab) as collab_count,
                 a.name as name
            RETURN node_id, paper_count, collab_count, name
            LIMIT $1
            """
            try:
                prolific = await self.store._run_query_method(query_prolific, [max_authors])
                all_authors.extend(prolific)
                print(f"      Found {len(prolific)} authors")
            except Exception as e:
                print(f"      Failed: {e}")
        
        if not all_authors:
            print("    No authors found!")
            return
        
        for author in all_authors:
            node_id = author.get('author_id') or author.get('node_id')
            if not node_id:
                continue
            
            paper_count = author.get('paper_count', 0) or 0
            collab_count = author.get('collab_count', 0) or 0
            
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
            
            if collab_count >= 100:
                collab_tier = 5
            elif collab_count >= 50:
                collab_tier = 4
            elif collab_count >= 20:
                collab_tier = 3
            elif collab_count >= 10:
                collab_tier = 2
            else:
                collab_tier = 1
            
            comm_id = f"A_{prod_tier}_{collab_tier}"
            self.author_communities[node_id] = comm_id
        
        author_counter = Counter(self.author_communities.values())
        self.author_community_sizes = dict(author_counter)
        
        print(f"\nCreated {len(set(self.author_communities.values()))} author communities")
        print(f"Covered {len(self.author_communities):,} authors")
        
        top_author_comms = sorted(self.author_community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 author communities:")
        for i, (comm_id, size) in enumerate(top_author_comms, 1):
            print(f"    {i}. {comm_id}: {size:,} authors")

    def _save_cache(self):
        cache_data = {
            'paper_communities': self.paper_communities,
            'author_communities': self.author_communities,
            'paper_community_sizes': self.paper_community_sizes,
            'author_community_sizes': self.author_community_sizes
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"\nUnified communities cached to {self.cache_file}")

    def load_cache(self) -> bool:
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
            print(f"Loaded unified communities from cache")
            print(f"Papers: {len(self.paper_communities):,} in {len(set(self.paper_communities.values()))} communities")
            print(f"Authors: {len(self.author_communities):,} in {len(set(self.author_communities.values()))} communities")
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def get_community(self, node_id: str, node_type: str = None) -> Optional[str]:
        if node_type == "paper" or node_id in self.paper_communities:
            return self.paper_communities.get(node_id)
        elif node_type == "author" or node_id in self.author_communities:
            return self.author_communities.get(node_id)
        else:
            return self.paper_communities.get(node_id) or self.author_communities.get(node_id)

    def get_community_size(self, community_id: str) -> int:
        if community_id.startswith('P_') or community_id.startswith('L_'):
            return self.paper_community_sizes.get(community_id, 0)
        elif community_id.startswith('A_'):
            return self.author_community_sizes.get(community_id, 0)
        else:
            return 0

    def get_statistics(self) -> Dict:
        return {
            'num_paper_communities': len(set(self.paper_communities.values())),
            'num_author_communities': len(set(self.author_communities.values())),
            'num_papers': len(self.paper_communities),
            'num_authors': len(self.author_communities),
            'avg_paper_community_size': np.mean(list(self.paper_community_sizes.values())) if self.paper_community_sizes else 0,
            'avg_author_community_size': np.mean(list(self.author_community_sizes.values())) if self.author_community_sizes else 0,
        }

async def build_and_cache_unified_communities(use_leiden: bool = True):
    print("Unified Community Detection")
    store = EnhancedStore()
    detector = CommunityDetector(store, use_leiden=use_leiden)
    
    if detector.load_cache():
        print("\n✓ Communities already cached!")
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo cache found, building communities...")
        await detector.build_communities(max_papers=None, max_authors=20000)
        stats = detector.get_statistics()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    await store.pool.close()
    print("\n✓ Done!")

if __name__ == "__main__":
    import sys
    use_leiden = "--leiden" in sys.argv or "-l" in sys.argv
    asyncio.run(build_and_cache_unified_communities(use_leiden=use_leiden))
