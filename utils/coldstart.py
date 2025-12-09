import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class ColdStartHandler:
    """Reward new papers based on content, not citations."""
    
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.author_reputation_cache = {}
    
    async def get_cold_start_reward(
        self, 
        paper_id: str,
        paper_node: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for papers with few/no citations.
        Uses content-based signals instead.
        """
        reward = 0.0
        
        # 1. Venue reputation 
        venue = paper_node.get('venue', '')
        if venue:
            venue_lower = venue.lower()
            if any(top.lower() in venue_lower for top in self.config.TOP_VENUES):
                reward += 1.5
            elif 'conference' in venue_lower or 'journal' in venue_lower:
                reward += 0.5
        
        try:
            authors = await self.store.get_authors_by_paper_id(paper_id)
            if authors:
                author_scores = []
                for author in authors:
                    author_id = author['author_id']
                    score = await self._get_author_reputation(author_id)
                    author_scores.append(score)
                
                max_author_score = max(author_scores) if author_scores else 0
                reward += min(max_author_score, 2.0)
        except:
            pass
        
        # 3. Recency boost
        year = paper_node.get('year')
        if year:
            current_year = datetime.now().year
            if year >= current_year:
                reward += 1.5
            elif year >= current_year - 1:
                reward += 1.0 
            elif year >= current_year - 2:
                reward += 0.5 
        
        # 4. Keyword match
        keywords = paper_node.get('fieldsOfStudy', '')
        if keywords:
            # Trending keywords
            trending = ['transformer', 'llm', 'diffusion', 'gnn', 'rl', 
                       'multimodal', 'federated', 'quantum']
            if any(trend in keywords.lower() for trend in trending):
                reward += 0.5
        
        return reward
    
    async def _get_author_reputation(self, author_id: str) -> float:
        """Calculate author reputation score."""
        # Check cache
        if author_id in self.author_reputation_cache:
            return self.author_reputation_cache[author_id]
        
        score = 0.0
        
        try:
            # Get author's papers
            papers = await self.store.get_papers_by_author_id(author_id)
            paper_count = len(papers)
            
            # Paper count contribution
            score += min(paper_count / 50.0, 1.0)
            
            # H-index contribution
            h_index = await self.store.get_author_h_index(author_id)
            score += min(h_index / 30.0, 1.0)
            
            # Affiliation contribution
            author_node = await self.store.get_author_by_id(author_id)
            if author_node:
                affiliation = author_node.get('affiliation', '')
                if any(inst in affiliation for inst in self.config.TOP_INSTITUTIONS):
                    score += 0.5
        
        except Exception as e:
            score = 0.0
        
        # Cache result
        self.author_reputation_cache[author_id] = score
        return score
    
    def is_cold_start(self, paper_node: Dict[str, Any]) -> bool:
        """Check if paper needs cold-start handling."""
        # Paper is cold-start if:
        # 1. Published in last 2 years, OR
        # 2. Has <10 citations, OR
        # 3. Authors have low h-index
        
        year = paper_node.get('year', 2000)
        current_year = datetime.now().year
        
        is_recent = (current_year - year) <= 2
        return is_recent
