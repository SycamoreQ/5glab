import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

try:
    from graph.database.comm_det import CommunityDetector
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False
    print("⚠ Community detection not available")


class RelationType:
    CITES = 0
    CITED_BY = 1
    WROTE = 2
    AUTHORED = 3
    SELF = 4
    COLLAB = 5
    KEYWORD_JUMP = 6
    VENUE_JUMP = 7
    OLDER_REF = 8
    NEWER_CITED_BY = 9
    SECOND_COLLAB = 10
    STOP = 11
    INFLUENCE_PATH = 12


class CommunityAwareRewardConfig:
    INTENT_MATCH_REWARD = 2.0
    INTENT_MISMATCH_PENALTY = -0.5
    DIVERSITY_BONUS = 0.5
    
    SEMANTIC_WEIGHT = 1.0
    NOVELTY_BONUS = 0.3
    DEAD_END_PENALTY = -1.0
    REVISIT_PENALTY = -0.8
    HIGH_DEGREE_BONUS = 0.3
    CITATION_COUNT_BONUS = 0.2
    RECENCY_BONUS = 0.2
    PROGRESS_REWARD = 0.5
    STAGNATION_PENALTY = -0.3
    GOAL_REACHED_BONUS = 5.0
    
    COMMUNITY_SWITCH_BONUS = 0.8       # Bonus for jumping to different community
    COMMUNITY_STUCK_PENALTY = -0.5     # Penalty per step stuck in same community
    COMMUNITY_LOOP_PENALTY = -1.0      # Severe penalty for returning to previous community
    DIVERSE_COMMUNITY_BONUS = 0.3      # Bonus for visiting many unique communities
    TEMPORAL_JUMP_BONUS = 0.2        # Bonus for moving to a community which is recent 
    TEMPORAL_JUMP_PENALTY = -0.1        # Bonus for doing the opposite 
    COMMUNITY_SIZE_BONUS = 0.3         # Bonus for not being in small communities
    BRIDGE_NODE_BONUS = 0.5 
    
    STUCK_THRESHOLD = 3                # Steps in same community = "stuck"
    SEVERE_STUCK_THRESHOLD = 5         # Very stuck threshold
    SEVERE_STUCK_MULTIPLIER = 2.0      # Multiply penalty when severely stuck


class AdvancedGraphTraversalEnv:
    """
    Community-aware hierarchical RL environment.
    Tracks and penalizes agent for getting stuck in local graph communities.
    """
    
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2", 
                 use_communities=True):
        self.store = store
        self.encoder = SentenceTransformer(embedding_model_name)
        self.query_embedding = None
        self.current_intent = None
        self.current_node = None
        self.visited = set()
        self.max_steps = 5
        self.current_step = 0
        self.text_dim = 384
        self.intent_dim = 5
        self.state_dim = self.text_dim * 2 + self.intent_dim
        
        self.pending_manager_action = None
        self.available_worker_nodes = []
        
        self.config = CommunityAwareRewardConfig()
        self.trajectory_history = []
        self.relation_types_used = set()
        self.best_similarity_so_far = -1.0
        self.previous_node_embedding = None
        
        self.use_communities = use_communities and COMMUNITY_AVAILABLE
        self.community_detector = None
        
        if self.use_communities:
            self.community_detector = CommunityDetector(store)
            if not self.community_detector.load_cache():
                print("No community cache found. Run: python -m RL.community_detection")
                print("Continuing without community rewards...")
                self.use_communities = False
        
        self.current_community = None
        self.community_history = []  
        self.community_visit_count = Counter()  
        self.comm_influence = {}
        self.steps_in_current_community = 0
        self.previous_community = None
        
        self.episode_stats = {
            'total_nodes_explored': 0,
            'unique_relation_types': 0,
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_achieved': 0.0,
            'unique_communities_visited': 0,
            'community_switches': 0,
            'max_steps_in_community': 0,
            'community_loops': 0
        }

    def _normalize_node_keys(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes Neo4j query results."""
        if not node:
            return {}
        normalized = {}
        for key, value in node.items():
            clean_key = key.split('.')[-1] if '.' in key else key
            normalized[clean_key] = value
        return normalized

    def _get_intent_vector(self, intent_enum: int):
        vec = np.zeros(self.intent_dim)
        if intent_enum < self.intent_dim:
            vec[intent_enum] = 1.0
        return vec


    def _update_community_tracking(self, node_id: str):
        if not self.use_communities:
            return
        
        new_community = self.community_detector.get_community(node_id)
        
        if new_community is None:
            if self.current_community is not None: 
                self.previous_community = self.current_community
                self.current_community = None 
                self.steps_in_current_community = 0 
            return
        
        if new_community != self.current_community:
            self.episode_stats['community_switches'] += 1
            
            if new_community in [c for _, c in self.community_history[:-1]]:
                self.episode_stats['community_loops'] += 1
            
            self.previous_community = self.current_community
            self.current_community = new_community
            self.steps_in_current_community = 1
        else:
            self.steps_in_current_community += 1
            self.episode_stats['max_steps_in_community'] = max(
                self.episode_stats['max_steps_in_community'],
                self.steps_in_current_community
            )
        
        self.community_history.append((self.current_step, new_community))
        self.community_visit_count[new_community] += 1
        self.episode_stats['unique_communities_visited'] = len(self.community_visit_count)


    def _calculate_community_reward(self) -> Tuple[float, str]:
        """
        Calculate reward/penalty based on community exploration patterns.
        Returns: (reward, reason)
        """
        if not self.use_communities or self.current_community is None:
            return 0.0, "no_community"
        
        if self.current_community is None:
            if self.previous_community is not None:
                return -0.1, "left_cache"
            return 0.0, "no_community"
        
        if len(self.community_history) < 2:
            return 0.0  , "no temporal history recorded"
    
        reward = 0.0
        reasons = []

        
        if self.steps_in_current_community == 1 and self.previous_community is not None:
            if self.current_community != self.previous_community:
                reward += self.config.COMMUNITY_SWITCH_BONUS
                reasons.append(f"switch_bonus:+{self.config.COMMUNITY_SWITCH_BONUS:.2f}")
        
        if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
            penalty = self.config.COMMUNITY_STUCK_PENALTY
            
            if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                reasons.append(f"severe_stuck:{penalty:.2f}")
            else:
                reasons.append(f"stuck:{penalty:.2f}")
            
            reward += penalty
        
        if len(self.community_history) > 1:
            previous_communities = [c for _, c in self.community_history[:-1]]
            if self.current_community in previous_communities:
                reward += self.config.COMMUNITY_LOOP_PENALTY
                reasons.append(f"loop:{self.config.COMMUNITY_LOOP_PENALTY:.2f}")

            if self.current_community: 
                visit_count = self.community_visit_count[self.current_community]
                
                if visit_count > 2: 
                    reward += self.config.REVISIT_PENALTY*(visit_count - 2)
                    reasons.append(f"revisit penalty{self.config.REVISIT_PENALTY:.2f}")
        
        unique_communities = len(self.community_visit_count)
        if unique_communities >= 3:
            diversity_bonus = self.config.DIVERSE_COMMUNITY_BONUS * (unique_communities - 2)
            reward += diversity_bonus
            reasons.append(f"diversity:+{diversity_bonus:.2f}")


        size = self.community_detector.get_community_size(self.current_community)
        
        if 5 <= size <= 50: 
            reward += self.config.COMMUNITY_SIZE_BONUS
            reasons.append(f"size:{self.config.COMMUNITY_SIZE_BONUS:.2f}")

        elif 50 <= size <= 200: 
            reward += 0.1 

        prev_comm = self.community_history[-2][1]
        
        try: 
            prev_year = int(prev_comm.split('_')[0])
            curr_year = int(self.current_community.split('_')[0])

            if prev_year < curr_year: 
                reward += self.config.TEMPORAL_JUMP_BONUS
            else: 
                reward += self.config.TEMPORAL_JUMP_PENALTY
        
        except: 
            pass 
                
        reason_str = ", ".join(reasons) if reasons else "none"
        return reward, reason_str


    async def reset(self, query: str, intent: int, start_node_id: str = None):
        """Reset environment with community tracking."""
        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
        self.current_step = 0
        self.pending_manager_action = None
        self.available_worker_nodes = []
    
        self.trajectory_history = []
        self.relation_types_used = set()
        self.best_similarity_so_far = -1.0
        self.previous_node_embedding = None
        
        self.current_community = None
        self.community_history = []
        self.community_visit_count = Counter()
        self.steps_in_current_community = 0
        self.previous_community = None
        
        self.episode_stats = {
            'total_nodes_explored': 0,
            'unique_relation_types': 0,
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_achieved': 0.0,
            'unique_communities_visited': 0,
            'community_switches': 0,
            'max_steps_in_community': 0,
            'community_loops': 0
        }
        
        if start_node_id:
            node = await self.store.get_paper_by_id(start_node_id)
            if not node:
                raise ValueError(f"Paper with ID '{start_node_id}' not found.")
            self.current_node = self._normalize_node_keys(node)
        else:
            candidates = await self.store.get_paper_by_title(query)
            if not candidates:
                raise ValueError(f"No starting node found for query: '{query}'")
            self.current_node = self._normalize_node_keys(candidates[0])

        if not self.current_node:
            raise ValueError(f"Failed to initialize current_node")
        
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        if not paper_id and not author_id:
            raise ValueError(f"Node has neither paper_id nor author_id")
        
        if paper_id:
            self.visited.add(paper_id)
            self._update_community_tracking(paper_id)
        if author_id:
            self.visited.add(author_id)
            self._update_community_tracking(author_id)
        
        self.previous_node_embedding = await self._get_node_embedding(self.current_node)
        
        return await self._get_state()

    async def _get_node_embedding(self, node: Dict[str, Any]) -> np.ndarray:
        """Get embedding for a node."""
        title = node.get('title', '')
        name = node.get('name', '')
        doi = node.get('doi', '')
        original_id = node.get('original_id', '')
        affiliation = node.get('affiliation', '')
        
        node_text = f"{title} {name} {doi} {original_id} {affiliation}".strip()
        
        if not node_text:
            node_text = node.get('paper_id') or node.get('author_id') or 'unknown_node'
        
        return self.encoder.encode(node_text if node_text else "empty")

    async def _get_state(self):
        """Constructs state vector."""
        node_emb = await self._get_node_embedding(self.current_node)
        intent_vec = self._get_intent_vector(self.current_intent)
        return np.concatenate([self.query_embedding, node_emb, intent_vec])

    async def get_manager_actions(self) -> List[int]:
        """Manager decides which relation type to explore."""
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        valid_relations = []
        
        if paper_id:
            refs = await self.store.get_references_by_paper(paper_id)
            if refs:
                valid_relations.append(RelationType.CITES)
            
            cites = await self.store.get_citations_by_paper(paper_id)
            if cites:
                valid_relations.append(RelationType.CITED_BY)
            
            authors = await self.store.get_authors_by_paper_id(paper_id)
            if authors:
                valid_relations.append(RelationType.WROTE)
            
            keywords = self.current_node.get('keywords', '')
            if keywords:
                valid_relations.append(RelationType.KEYWORD_JUMP)
            
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue:
                valid_relations.append(RelationType.VENUE_JUMP)
            
            older = await self.store.get_older_references(paper_id)
            if older:
                valid_relations.append(RelationType.OLDER_REF)
            
            newer = await self.store.get_newer_citations(paper_id)
            if newer:
                valid_relations.append(RelationType.NEWER_CITED_BY)
        
        elif author_id:
            papers = await self.store.get_papers_by_author_id(author_id)
            if papers:
                valid_relations.append(RelationType.AUTHORED)
            
            collabs = await self.store.get_collabs_by_author(author_id)
            if collabs:
                valid_relations.append(RelationType.COLLAB)
            
            second_collabs = await self.store.get_second_degree_collaborators(author_id)
            if second_collabs:
                valid_relations.append(RelationType.SECOND_COLLAB)
            
            influence = await self.store.get_influence_path_papers(author_id)
            if influence:
                valid_relations.append(RelationType.INFLUENCE_PATH)
        
        valid_relations.append(RelationType.STOP)
        
        return valid_relations

    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        """Manager step with standard rewards."""
        self.pending_manager_action = relation_type
        
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps

            if self.previous_node_embedding is None: 
                return True , -1.0 
            current_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            
            if current_sim > 0.7:
                return True, self.config.GOAL_REACHED_BONUS
            elif current_sim > 0.4:
                return True, 0.5
            else:
                return True, -1.0
        
        manager_reward = 0.0
        
        if relation_type == self.current_intent:
            manager_reward += self.config.INTENT_MATCH_REWARD
        else:
            manager_reward += self.config.INTENT_MISMATCH_PENALTY
        
        if relation_type not in self.relation_types_used:
            manager_reward += self.config.DIVERSITY_BONUS
            self.relation_types_used.add(relation_type)
            self.episode_stats['unique_relation_types'] += 1
        
        # Fetch nodes (same as before)
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        raw_nodes = []

        if relation_type == RelationType.CITES and paper_id:
            raw_nodes = await self.store.get_references_by_paper(paper_id)
        elif relation_type == RelationType.CITED_BY and paper_id:
            raw_nodes = await self.store.get_citations_by_paper(paper_id)
        elif relation_type == RelationType.WROTE and paper_id:
            raw_nodes = await self.store.get_authors_by_paper_id(paper_id)
        elif relation_type == RelationType.AUTHORED and author_id:
            raw_nodes = await self.store.get_papers_by_author_id(author_id)
        elif relation_type == RelationType.COLLAB and author_id:
            raw_nodes = await self.store.get_collabs_by_author(author_id)
        elif relation_type == RelationType.KEYWORD_JUMP and paper_id:
            keywords = self.current_node.get('keywords', '')
            if keywords:
                keyword = keywords.split(',')[0].strip() if isinstance(keywords, str) else str(keywords[0])
                raw_nodes = await self.store.get_papers_by_keyword(keyword, limit=5, exclude_paper_id=paper_id)
        elif relation_type == RelationType.VENUE_JUMP and paper_id:
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue:
                raw_nodes = await self.store.get_papers_by_venue(venue, exclude_paper_id=paper_id)
        elif relation_type == RelationType.OLDER_REF and paper_id:
            raw_nodes = await self.store.get_older_references(paper_id)
        elif relation_type == RelationType.NEWER_CITED_BY and paper_id:
            raw_nodes = await self.store.get_newer_citations(paper_id)
        elif relation_type == RelationType.SECOND_COLLAB and author_id:
            raw_nodes = await self.store.get_second_degree_collaborators(author_id)
        elif relation_type == RelationType.INFLUENCE_PATH and author_id:
            raw_nodes = await self.store.get_influence_path_papers(author_id)
        
        
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]
        self.available_worker_nodes = [(node, relation_type) for node in normalized_nodes]
        self.available_worker_nodes = [
            (node, r_type) for node, r_type in self.available_worker_nodes 
            if not self._is_visited(node)
        ]
        
        num_available = len(self.available_worker_nodes)
        if num_available == 0:
            manager_reward += self.config.DEAD_END_PENALTY
            self.episode_stats['dead_ends_hit'] += 1
        elif num_available > 10:
            manager_reward += self.config.HIGH_DEGREE_BONUS
        
        return False, manager_reward

    def _is_visited(self, node: Dict[str, Any]) -> bool:
        """Check if visited."""
        paper_id = node.get('paper_id')
        author_id = node.get('author_id')
        return (paper_id and paper_id in self.visited) or (author_id and author_id in self.visited)

    async def get_worker_actions(self) -> List[Tuple[Dict, int]]:
        """Get worker actions."""
        return self.available_worker_nodes

    async def worker_step(self, chosen_node: Dict) -> Tuple[np.ndarray, float, bool]:
        """
        Worker step with COMMUNITY-AWARE rewards.
        """
        self.current_step += 1
        self.episode_stats['total_nodes_explored'] += 1
        
        self.current_node = self._normalize_node_keys(chosen_node)
        
        # Check revisit
        is_revisit = False
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        if paper_id and paper_id in self.visited:
            is_revisit = True
            self.episode_stats['revisits'] += 1
        if author_id and author_id in self.visited:
            is_revisit = True
            self.episode_stats['revisits'] += 1
        
        if paper_id:
            self.visited.add(paper_id)
            self._update_community_tracking(paper_id)
        if author_id:
            self.visited.add(author_id)
            self._update_community_tracking(author_id)
        
        worker_reward = 0.0
        
        # Semantic similarity
        node_emb = await self._get_node_embedding(self.current_node)
        semantic_sim = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
        worker_reward += semantic_sim * self.config.SEMANTIC_WEIGHT
        
        if semantic_sim > self.best_similarity_so_far:
            self.best_similarity_so_far = semantic_sim
            self.episode_stats['max_similarity_achieved'] = semantic_sim
        
        # Progress reward
        if self.previous_node_embedding is not None:
            prev_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            if semantic_sim > prev_sim + 0.05:
                worker_reward += self.config.PROGRESS_REWARD
            elif semantic_sim < prev_sim - 0.1:
                worker_reward += self.config.STAGNATION_PENALTY
        
        # Novelty/revisit
        if not is_revisit and semantic_sim > 0.5:
            worker_reward += self.config.NOVELTY_BONUS
        if is_revisit:
            worker_reward += self.config.REVISIT_PENALTY
        
        # Citation bonus
        if paper_id:
            citation_count = await self.store.get_citation_count(paper_id)
            if citation_count > 100:
                worker_reward += self.config.CITATION_COUNT_BONUS * np.log10(citation_count / 100)
        
        # Recency bonus
        year = self.current_node.get('year')
        if year and year >= 2020:
            worker_reward += self.config.RECENCY_BONUS
        
        community_reward, community_reason = self._calculate_community_reward()
        worker_reward += community_reward

                    
        self.previous_node_embedding = node_emb
        self.trajectory_history.append({
            'node': self.current_node,
            'similarity': semantic_sim,
            'reward': worker_reward,
            'community': self.current_community,
            'community_reward': community_reward,
            'community_reason': community_reason
        })
        
        done = self.current_step >= self.max_steps
        
        if done and semantic_sim > 0.7:
            worker_reward += self.config.GOAL_REACHED_BONUS * semantic_sim
        
        next_state = await self._get_state()
        
        self.pending_manager_action = None
        self.available_worker_nodes = []
        
        return next_state, worker_reward, done

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get detailed episode summary with community stats."""
        return {
            **self.episode_stats,
            'path_length': len(self.trajectory_history),
            'relation_diversity': len(self.relation_types_used),
            'community_diversity_ratio': (
                self.episode_stats['unique_communities_visited'] / 
                max(1, self.episode_stats['total_nodes_explored'])
            )
        }

    async def get_valid_actions(self) -> List[Tuple[Dict, int]]:
        paper_id = self.current_node.get('paper_id')
        refs = await self.store.get_references_by_paper(paper_id)
        actions = [(self._normalize_node_keys(n), RelationType.CITES) for n in refs]
        cites = await self.store.get_citations_by_paper(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.CITED_BY) for n in cites]
        authors = await self.store.get_authors_by_paper_id(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.WROTE) for n in authors]
        return [(node, r_type) for node, r_type in actions if not self._is_visited(node)]

    async def step(self, action_node: Dict, action_relation: int):
        self.current_step += 1
        self.current_node = self._normalize_node_keys(action_node)
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        if paper_id:
            self.visited.add(paper_id)
        if author_id:
            self.visited.add(author_id)
        struct_reward = 1.0 if action_relation == self.current_intent else -0.5
        node_emb = await self._get_node_embedding(self.current_node)
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
        total_reward = (0.6 * struct_reward) + (0.4 * sem_reward)
        done = self.current_step >= self.max_steps
        return await self._get_state(), total_reward, done