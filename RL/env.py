import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque , Counter
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
    CITES = 0           # Paper: Get references (outgoing)
    CITED_BY = 1        # Paper: Get citations (incoming)
    WROTE = 2           # Paper -> Author
    AUTHORED = 3        # Author -> Paper
    SELF = 4            # The node itself
    COLLAB = 5          # Author -> Co-Authors
    KEYWORD_JUMP = 6    # Paper: Similar Papers (via Keyword)
    VENUE_JUMP = 7      # Paper: Similar Papers (via Venue/Publication)
    OLDER_REF = 8       # Paper: Older References
    NEWER_CITED_BY = 9  # Paper: Newer Citations
    SECOND_COLLAB = 10  # Author: 2nd Degree Collaborators
    STOP = 11           # Stops further exploration 
    INFLUENCE_PATH = 12 


class RewardPenalty: 
    INTENT_MATCH =  2.0
    INTENT_MISMATCH = -0.5
    DIVERSITY_BONUS =  0.5

    SEMANTIC_WEIGHT = 1.0              # Weight for semantic similarity
    NOVELTY_BONUS = 0.3                # Bonus for discovering new nodes
    DEAD_END_PENALTY = -1.0            # Penalty for nodes with no neighbors
    REVISIT_PENALTY = -0.8             # Penalty for returning to visited nodes
    
    HIGH_DEGREE_BONUS = 0.3            # Bonus for well-connected nodes
    CITATION_COUNT_BONUS = 0.2         # Bonus for highly cited papers
    RECENCY_BONUS = 0.2                # Bonus for recent papers (if year available)
    COLLAB_COUNT_BONUS = 0.2
    NEWER_CITATION_BONUS = 0.2
    
    PROGRESS_REWARD = 0.5              # Getting closer to query goal
    STAGNATION_PENALTY = -0.3          # Not making progress
    
    PATH_EFFICIENCY_BONUS = 1.0        # Bonus for efficient paths
    GOAL_REACHED_BONUS = 5.0           # Large bonus for reaching goal state
    
    EXPLORATION_BONUS = 0.2            # Early episodes: encourage exploration
    EXPLOITATION_BONUS = 0.4           # Later episodes: encourage exploitation

    COMMUNITY_SWITCH_BONUS = 0.8       # Bonus for jumping to different community
    COMMUNITY_STUCK_PENALTY = -0.5     # Penalty per step stuck in same community
    COMMUNITY_LOOP_PENALTY = -1.0      # Severe penalty for returning to previous community
    DIVERSE_COMMUNITY_BONUS = 0.3      # Bonus for visiting many unique communities

    STUCK_THRESHOLD = 3 
    SEVERE_STUCK_THRESHOLD = 5 
    SEVERE_STUCK_MULTIPLIER = 2.0 
    

class AdvancedGraphTraversalEnv:
    """
    Goal-Conditioned Hierarchical RL Environment.
    Manager chooses relation type, Worker chooses specific node.
    """
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2" , use_communities = True):
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

        self.config = RewardPenalty()
        self.trajectory_history = []  # For path efficiency calculation
        self.relation_types_used = set()  # For diversity bonus
        self.best_similarity_so_far = -1.0  # For progress tracking
        self.previous_node_embedding = None  # For progress calculation

        self.use_communities = use_communities and COMMUNITY_AVAILABLE 

        if self.use_communities: 
            self.community_detector = CommunityDetector(store)

            if not self.community_detector.load_cache():
                print("no community cache found")
                print("continuing without community rewards")
                self.use_communities = False 


        self.current_community = None 
        self.community_history = []
        self.community_visit_count = Counter()
        self.steps_in_curr_comm = 0 
        self.prev_community = None 

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
    

    def _calculate_community_rewards(self) -> float: 
        if not self.use_communities or self.current_community is None: 
            return 0.0
        

        if len(self.current_community < 10): 
            self.use_communities = False
            print("Small communitites can be avoided")
            
        self.use_communities = True

        comm_reward = 0.0 

        if self.prev_community and self.current_community != self.prev_community:
            comm_reward += self.config.COMMUNITY_SWITCH_BONUS
        
        if self.steps_in_curr_comm >= self.config.STUCK_THRESHOLD: 
            penalty = self.config.COMMUNITY_STUCK_PENALTY
            
            if self.steps_in_curr_comm >= self.config.SEVERE_STUCK_THRESHOLD: 
                penalty *= self.config.SEVERE_STUCK_MULTIPLIER 
                
            else: 
                print("stuck")
            
            comm_reward += penalty 


        if (self.prev_community and self.current_community in [c for _ , c in self.community_history[:,2]]):
            comm_reward += self.config.COMMUNITY_LOOP_PENALTY


        unique_comms = len(self.community_visit_count)
        if unique_comms >= 3: 
            diversity_bonus = self.config.DIVERSE_COMMUNITY_BONUS * (unique_comms - 2)
            comm_reward += diversity_bonus

        
        return comm_reward 
    
  

    async def reset(self, query: str, intent: int, start_node_id: str = None):
        """
        Reset environment with a query and intent.
        """
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
                raise ValueError(f"Paper with ID '{start_node_id}' not found in database.")
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
            raise ValueError(f"Node has neither paper_id nor author_id: {self.current_node}")
        
        if paper_id:
            self.visited.add(paper_id)
        if author_id:
            self.visited.add(author_id)
            
        return await self._get_state()
    

    async def _get_state(self):
        """
        Constructs state vector: [Query Embedding (384) | Node Embedding (384) | Intent (5)]
        """
        title = self.current_node.get('title', '')
        name = self.current_node.get('name', '')
        doi = self.current_node.get('doi', '')
        original_id = self.current_node.get('original_id', '')
        
        node_text = f"{title} {name} {doi} {original_id}".strip()
        
        if not node_text or node_text == '':
            node_text = self.current_node.get('paper_id', 'unknown_node')
        
        node_emb = self.encoder.encode(node_text if node_text else "empty")
        intent_vec = self._get_intent_vector(self.current_intent)
        
        return np.concatenate([self.query_embedding, node_emb, intent_vec])
    

    async def get_manager_actions(self) -> List[int]:
        """
        Manager decides which relation type to explore.
        Returns list of valid RelationType values.
        """
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        valid_relations = []
        
        if paper_id:
            
            # CITES (outgoing references)
            refs = await self.store.get_references_by_paper(paper_id)
            if refs:
                valid_relations.append(RelationType.CITES)
            
            # CITED_BY (incoming citations)
            cites = await self.store.get_citations_by_paper(paper_id)
            if cites:
                valid_relations.append(RelationType.CITED_BY)
            
            # WROTE (paper -> authors)
            authors = await self.store.get_authors_by_paper_id(paper_id)
            if authors:
                valid_relations.append(RelationType.WROTE)
            
            # KEYWORD_JUMP
            keywords = self.current_node.get('keywords', '')
            if keywords:
                valid_relations.append(RelationType.KEYWORD_JUMP)
            
            # VENUE_JUMP
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue:
                valid_relations.append(RelationType.VENUE_JUMP)
            
            # OLDER_REF
            older = await self.store.get_older_references(paper_id)
            if older:
                valid_relations.append(RelationType.OLDER_REF)
            
            # NEWER_CITED_BY
            newer = await self.store.get_newer_citations(paper_id)
            if newer:
                valid_relations.append(RelationType.NEWER_CITED_BY)
        
        elif author_id:
            # Author node - check author-specific actions
            
            # AUTHORED (author -> papers)
            papers = await self.store.get_papers_by_author_id(author_id)
            if papers:
                valid_relations.append(RelationType.AUTHORED)
            
            # COLLAB (author -> collaborators)
            collabs = await self.store.get_collabs_by_author(author_id)
            if collabs:
                valid_relations.append(RelationType.COLLAB)
            
            # SECOND_COLLAB 
            second_collabs = await self.store.get_second_degree_collaborators(author_id)
            if second_collabs:
                valid_relations.append(RelationType.SECOND_COLLAB)
            
            # INFLUENCE_PATH
            influence = await self.store.get_influence_path_papers(author_id)
            if influence:
                valid_relations.append(RelationType.INFLUENCE_PATH)
        
        valid_relations.append(RelationType.STOP)
        
        return valid_relations

    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        """
        Manager chooses a relation type.
        Returns: (is_terminal, manager_reward)
        """
        self.pending_manager_action = relation_type
        
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps 
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
            manager_reward += self.config.INTENT_MATCH
        else:
            manager_reward += self.config.INTENT_MISMATCH


        if relation_type not in self.relation_types_used: 
            manager_reward += self.config.DIVERSITY_BONUS
            self.relation_types_used.add(relation_type)
            self.episode_stats['unique_relation_types'] += 1 

                
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
        """Check if a node has been visited."""
        paper_id = node.get('paper_id')
        author_id = node.get('author_id')
        
        if paper_id and paper_id in self.visited:
            return True
        if author_id and author_id in self.visited:
            return True
        return False

    async def get_worker_actions(self) -> List[Tuple[Dict, int]]:
        """
        Worker gets list of available nodes for the manager's chosen relation.
        Returns: List of (node_dict, relation_type) tuples
        """
        return self.available_worker_nodes

    async def worker_step(self, chosen_node: Dict) -> Tuple[np.ndarray, float, bool]:
        """
        Worker chooses a specific node from available options.
        Returns: (next_state, worker_reward, done)
        """
        self.current_step += 1
        self.episode_stats['total_nodes_explored'] += 1 
        self.current_node = self._normalize_node_keys(chosen_node)

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
        if author_id:
            self.visited.add(author_id)

        
        worker_reward = 0.0 
      
        title = self.current_node.get('title', '')
        name = self.current_node.get('name', '')
        doi = self.current_node.get('doi', '')
        original_id = self.current_node.get('original_id', '')
        
        node_text = f"{title} {name} {doi} {original_id}".strip()
        
        if not node_text or node_text == '':
            node_text = self.current_node.get('paper_id', 'unknown_node')
        
        node_emb = self.encoder.encode(node_text if node_text else "empty")
        
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )

        worker_reward = sem_reward * self.config.SEMANTIC_WEIGHT
        
        if sem_reward > self.best_similarity_so_far: 
            self.best_similarity_so_far = sem_reward
            self.episode_stats['max_similarity_acheived'] = sem_reward

        
        if self.previous_node_embedding is not None: 
            prev_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            ) 

            if sem_reward > prev_sim + 0.05:
                worker_reward += self.config.PROGRESS_REWARD

            elif sem_reward < prev_sim - 0.1:
                worker_reward += self.config.STAGNATION_PENALTY
        
        if not is_revisit and sem_reward > 0.1:
            worker_reward += self.config.NOVELTY_BONUS

        if is_revisit: 
            worker_reward += self.config.REVISIT_PENALTY

        if paper_id: 
            citation_count = await self.store.get_citation_count(paper_id)

            if citation_count > 100: 
                worker_reward += self.config.CITATION_COUNT_BONUS * np.log10(citation_count / 100)


        if author_id: 
            collab_count = await self.store.get_collab_count(author_id)
            if collab_count > 50: 
                worker_reward += self.config.COLLAB_COUNT_BONUS * np.log10(collab_count/100)

        year = self.current_node.get('year')
        
        if year and year > 2020 and paper_id:
            new_citations = await self.store.get_newer_citations(paper_id)

            if new_citations > 100: 
                worker_reward += self.config.RECENCY_BONUS + self.config.NEWER_CITATION_BONUS

        self.previous_node_embedding = node_emb

        community_reward = self._calculate_community_rewards()
        worker_reward += community_reward
        
        self.previous_node_embedding = node_emb
        self.trajectory_history.append({
            'node': self.current_node,
            'similarity': sem_reward,
            'reward': worker_reward,
            'community': self.current_community,
            'community_reward': community_reward,
        })

        done = self.current_step >= self.max_steps
        if done and sem_reward > 0.7:
            worker_reward += self.config.GOAL_REACHED_BONUS * sem_reward
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
        """Legacy method - returns all possible (node, relation) pairs"""
        paper_id = self.current_node.get('paper_id')
        
        refs = await self.store.get_references_by_paper(paper_id)
        actions = [(self._normalize_node_keys(n), RelationType.CITES) for n in refs]
        
        cites = await self.store.get_citations_by_paper(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.CITED_BY) for n in cites]
        
        authors = await self.store.get_authors_by_paper_id(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.WROTE) for n in authors]

        valid_actions = [
            (node, r_type) for node, r_type in actions 
            if not self._is_visited(node)
        ]
        
        return valid_actions

    async def step(self, action_node: Dict, action_relation: int):
        """Legacy single-step method"""
        self.current_step += 1
        self.current_node = self._normalize_node_keys(action_node)
        
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        if paper_id:
            self.visited.add(paper_id)
        if author_id:
            self.visited.add(author_id)
        
        if action_relation == self.current_intent:
            struct_reward = 1.0
        else:
            struct_reward = -0.5
            
        node_text = self.current_node.get('title', '') or self.current_node.get('name', '')
        node_emb = self.encoder.encode(node_text if node_text else "empty")
        
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )

        total_reward = (0.6 * struct_reward) + (0.4 * sem_reward)
        
        done = self.current_step >= self.max_steps
        return await self._get_state(), total_reward, done