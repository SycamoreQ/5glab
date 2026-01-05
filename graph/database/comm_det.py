import asyncio
import pickle
import os
import hashlib
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


class CommunityDetector:
    def __init__(self, store=None, cache_file: str = None):
        if cache_file:
            self.cache_file = cache_file
        else:
            self.cache_file = 'community_cache_1M.pkl'
        self.store = store
        self.paper_communities: Dict[str, str] = {}
        self.paper_community_sizes: Dict[str, int] = {}
        self.author_communities: Dict[str, str] = {}
        self.author_community_sizes: Dict[str, int] = {}
        self.is_loaded = False

    def normalize_paper_id(self, paper_id: str) -> str:
        if not paper_id:
            return ""
        paper_id = str(paper_id).strip().lstrip('0')
        return paper_id if paper_id else "0"

    def normalize_author_id(self, author_id: str) -> str:
        if not author_id:
            return ""
        author_id = str(author_id).strip().lstrip('0')
        return author_id if author_id else "0"

    async def build_communities_from_cache(
        self,
        target_communities: int = 500,
        min_author_papers: int = 5
    ):
        print(f"BUILDING HASH-BASED COMMUNITIES (TARGET: {target_communities})")
        
        cache_dir = 'training_cache'
        papers_file = os.path.join(cache_dir, 'training_papers_1M.pkl')
        
        print(f"Loading papers from {papers_file}...")
        with open(papers_file, 'rb') as f:
            papers = pickle.load(f)
        
        paper_ids = [
            self.normalize_paper_id(str(p.get('paperId') or p.get('paper_id')))
            for p in papers
        ]
        paper_ids = [pid for pid in paper_ids if pid]
        
        print(f"Loaded {len(paper_ids):,} papers")
        
        await self.detect_paper_communities_hash_based(paper_ids, target_communities)
        await self._build_author_communities(papers, min_author_papers)
        
        self.save_cache()
        self.is_loaded = True
        
        print(f"All communities cached to {self.cache_file}")

    async def detect_paper_communities_hash_based(
        self,
        paper_ids: List[str],
        target_communities: int
    ):
        print("\nPAPER COMMUNITIES (HASH-BASED SHARDING)")
        
        print("Fetching metadata for enrichment...")
        metadata_map = await self._fetch_metadata_batch(paper_ids)
        
        print(f"\nAssigning papers to {target_communities} balanced communities...")
        
        paper_communities: Dict[str, str] = {}
        field_counter = defaultdict(lambda: defaultdict(int))
        
        for pid in tqdm(paper_ids, desc="Assignment"):
            meta = metadata_map.get(pid, {})
            
            field = self._extract_field_simple(meta)
            cite_tier = self._get_citation_tier(meta.get('cites', 0))
            
            hash_val = int(hashlib.md5(pid.encode()).hexdigest(), 16)
            shard_id = hash_val % target_communities
            
            comm_id = f"C{shard_id:04d}_{field}_{cite_tier}"
            paper_communities[pid] = comm_id
            
            field_counter[field][cite_tier] += 1
        
        self.paper_communities = paper_communities
        self.paper_community_sizes = dict(Counter(self.paper_communities.values()))
        
        self._print_distribution(field_counter)
        self._print_paper_distribution()

    def _get_citation_tier(self, cites: int) -> str:
        if cites >= 100:
            return 'H'
        elif cites >= 20:
            return 'M'
        elif cites >= 5:
            return 'L'
        else:
            return 'VL'

    def _extract_field_simple(self, meta: Dict) -> str:
        fields = meta.get('fields', []) or []
        venue = str(meta.get('venue', '')).lower()
        title = str(meta.get('title', '')).lower()
        
        if isinstance(fields, list) and fields:
            first = fields[0]
            if isinstance(first, dict):
                field = str(first.get('category', ''))
            elif isinstance(first, str):
                field = str(first)
            else:
                field = ''
            
            if field and len(field) > 2:
                return field[:6].upper().replace(' ', '')
        
        text = (venue + ' ' + title).lower()
        
        keywords = {
            'CS': ['comput', 'algorithm', 'software'],
            'AI': ['machine learn', 'deep learn', 'neural', 'ai'],
            'MED': ['medic', 'clinic', 'patient', 'health'],
            'BIO': ['biolog', 'gene', 'protein', 'cell'],
            'PHY': ['physic', 'quantum', 'particle'],
            'CHEM': ['chemis', 'molecul', 'reaction'],
            'ENG': ['engineer', 'material'],
            'MATH': ['mathematic', 'theorem'],
            'SOC': ['social', 'society'],
            'ECO': ['econom', 'financ'],
        }
        
        for field_name, kws in keywords.items():
            if any(kw in text for kw in kws):
                return field_name
        
        return 'GEN'

    async def _fetch_metadata_batch(self, paper_ids: List[str]) -> Dict[str, Dict]:
        metadata_map = {}
        batch_size = 5000
        
        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i:i + batch_size]
            
            query = """
            MATCH (p:Paper)
            WHERE p.paperId IN $1
            RETURN p.paperId as paper_id,
                   p.year as year,
                   COALESCE(p.citationCount, 0) as cites,
                   p.fieldsOfStudy as fields,
                   p.title as title,
                   p.venue as venue
            """
            
            batch_results = await self.store._run_query_method(query, [batch])
            
            for r in batch_results:
                pid = self.normalize_paper_id(str(r['paper_id']))
                metadata_map[pid] = r
            
            if (i + batch_size) % 20000 == 0:
                print(f"  Fetched {len(metadata_map):,}/{len(paper_ids):,} papers")
        
        return metadata_map

    def _print_distribution(self, field_counter):
        print("\nFIELD AND CITATION DISTRIBUTION:")
        
        for field in sorted(field_counter.keys()):
            total = sum(field_counter[field].values())
            print(f"\n  {field}:")
            for tier in ['H', 'M', 'L', 'VL']:
                count = field_counter[field].get(tier, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {tier}: {count:>8,} ({pct:>5.1f}%)")

    def _print_paper_distribution(self):
        comm_counts = self.paper_community_sizes
        total = len(self.paper_communities)
        
        print("\nCOMMUNITY SIZE DISTRIBUTION:")
        
        sizes = sorted(comm_counts.values())
        print(f"  Min:     {sizes[0]:>8,}")
        print(f"  Q1:      {sizes[len(sizes)//4]:>8,}")
        print(f"  Median:  {sizes[len(sizes)//2]:>8,}")
        print(f"  Q3:      {sizes[3*len(sizes)//4]:>8,}")
        print(f"  Max:     {sizes[-1]:>8,}")
        print(f"  Mean:    {np.mean(sizes):>8,.1f}")
        print(f"  Std:     {np.std(sizes):>8,.1f}")
        
        print(f"\nTop 30 Communities:")
        for comm, count in sorted(comm_counts.items(), key=lambda x: -x[1])[:30]:
            pct = 100 * count / total
            print(f"  {comm:<30} {count:>8,} ({pct:>5.1f}%)")
        
        print("\nSUMMARY:")
        print(f"  Total papers:         {total:>10,}")
        print(f"  Total communities:    {len(comm_counts):>10,}")
        print(f"  Avg size:             {total/len(comm_counts):>10,.1f}")
        print(f"  Balance ratio:        {sizes[-1]/sizes[0]:>10,.2f}x")
        print(f"\nCreated {len(comm_counts):,} balanced communities")

    async def _build_author_communities(
        self,
        papers: List[Dict],
        min_author_papers: int
    ):
        print("\nAUTHOR COMMUNITIES")
        
        author_stats = defaultdict(lambda: {"paper_count": 0, "coauthors": set()})
        
        for paper in tqdm(papers, desc="Extracting authors"):
            paper_authors = paper.get('authors', [])
            if not paper_authors:
                continue
            
            author_ids = []
            for author in paper_authors:
                author_id = author.get('authorId') or author.get('author_id')
                if author_id:
                    aid = self.normalize_author_id(str(author_id))
                    author_stats[aid]["paper_count"] += 1
                    author_ids.append(aid)
            
            for i, aid1 in enumerate(author_ids):
                for aid2 in author_ids[i + 1:]:
                    author_stats[aid1]["coauthors"].add(aid2)
                    author_stats[aid2]["coauthors"].add(aid1)
        
        print(f"Found {len(author_stats):,} authors")
        
        filtered_authors = {
            aid: stats
            for aid, stats in author_stats.items()
            if stats["paper_count"] >= min_author_papers
        }
        
        print(f"Kept {len(filtered_authors):,} authors")
        
        for author_id, stats in filtered_authors.items():
            paper_count = stats["paper_count"]
            collab_count = len(stats["coauthors"])
            
            prod_tier = (
                'P5' if paper_count >= 50 else
                'P4' if paper_count >= 20 else
                'P3' if paper_count >= 10 else
                'P2'
            )
            collab_tier = (
                'C5' if collab_count >= 100 else
                'C4' if collab_count >= 50 else
                'C3' if collab_count >= 20 else
                'C2'
            )
            
            self.author_communities[author_id] = f"A_{prod_tier}_{collab_tier}"
        
        self.author_community_sizes = dict(Counter(self.author_communities.values()))
        print(f"Created {len(set(self.author_communities.values()))} author communities")

    def save_cache(self):
        cache_data = {
            'paper_communities': self.paper_communities,
            'paper_community_sizes': self.paper_community_sizes,
            'author_communities': self.author_communities,
            'author_community_sizes': self.author_community_sizes,
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"\nCache saved to {self.cache_file}")

    def load_cache(self) -> bool:
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.paper_communities = cache_data.get('paper_communities', {})
            self.paper_community_sizes = cache_data.get('paper_community_sizes', {})
            self.author_communities = cache_data.get('author_communities', {})
            self.author_community_sizes = cache_data.get('author_community_sizes', {})
            self.is_loaded = True
            
            print(f"Loaded communities from {self.cache_file}")
            print(f"  Papers: {len(self.paper_communities):,} in {len(set(self.paper_communities.values())):,} communities")
            print(f"  Authors: {len(self.author_communities):,} in {len(set(self.author_communities.values())):,} communities")
            
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def get_community(self, node_id: str, node_type: str = None) -> Optional[str]:
        if node_type == 'paper' or not node_type:
            node_id_norm = self.normalize_paper_id(node_id)
            comm = self.paper_communities.get(node_id_norm)
            if comm:
                return comm
        
        if node_type == 'author' or not node_type:
            node_id_norm = self.normalize_author_id(node_id)
            return self.author_communities.get(node_id_norm)
        
        return None

    def get_community_size(self, community_id: str) -> int:
        if community_id in self.paper_community_sizes:
            return self.paper_community_sizes[community_id]
        if community_id in self.author_community_sizes:
            return self.author_community_sizes[community_id]
        return 0

    def get_statistics(self) -> Dict:
        paper_sizes = list(self.paper_community_sizes.values())
        author_sizes = list(self.author_community_sizes.values())
        
        return {
            'num_paper_communities': len(self.paper_community_sizes),
            'num_author_communities': len(self.author_community_sizes),
            'num_papers': len(self.paper_communities),
            'num_authors': len(self.author_communities),
            'avg_paper_community_size': float(np.mean(paper_sizes)) if paper_sizes else 0.0,
            'avg_author_community_size': float(np.mean(author_sizes)) if author_sizes else 0.0,
            'balance_ratio': float(max(paper_sizes) / min(paper_sizes)) if paper_sizes else 0.0,
        }


async def build_and_cache_communities(
    target_communities: int = 500,
    min_author_papers: int = 5
):
    from graph.database.store import EnhancedStore
    
    print("\nHASH-BASED COMMUNITY DETECTION")
    
    store = EnhancedStore()
    detector = CommunityDetector(store=store, cache_file='community_cache_1M.pkl')
    
    if detector.load_cache():
        print("\nCommunities already cached")
        stats = detector.get_statistics()
        print("\nStatistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value:,}")
        print("\nTo rebuild, delete 'community_cache_1M.pkl' and run again.")
        await store.pool.close()
        return
    
    print("Cache not found. Building communities...")
    await detector.build_communities_from_cache(
        target_communities=target_communities,
        min_author_papers=min_author_papers
    )
    
    await store.pool.close()
    print("\nDone")


if __name__ == '__main__':
    import sys
    
    target = 500
    min_papers = 5
    
    for i, arg in enumerate(sys.argv):
        if arg in ('--target', '-t') and i + 1 < len(sys.argv):
            target = int(sys.argv[i + 1])
        if arg in ('--min-papers', '-p') and i + 1 < len(sys.argv):
            min_papers = int(sys.argv[i + 1])
    
    print("Parameters:")
    print(f"  Target communities: {target}")
    print(f"  Min author papers: {min_papers}")
    
    asyncio.run(build_and_cache_communities(
        target_communities=target,
        min_author_papers=min_papers
    ))
