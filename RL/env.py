import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

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

class AdvancedGraphTraversalEnv:
    """
    Goal-Conditioned Hierarchical RL Environment.
    Manager chooses relation type, Worker chooses specific node.
    """
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2"):
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
        
        # Hierarchical RL state
        self.pending_manager_action = None
        self.available_worker_nodes = []

    def _normalize_node_keys(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes Neo4j query results to have consistent keys.
        Now store.py returns clean keys directly, but keep this for safety.
        """
        if not node:
            return {}
        
        normalized = {}
        for key, value in node.items():
            # Remove prefixes like 'p.', 'ref.', 'a.', 'citing.', etc.
            clean_key = key.split('.')[-1] if '.' in key else key
            normalized[clean_key] = value
        return normalized

    def _get_intent_vector(self, intent_enum: int):
        vec = np.zeros(self.intent_dim)
        if intent_enum < self.intent_dim:
            vec[intent_enum] = 1.0
        return vec

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
        
        if start_node_id:
            # Use provided paper_id
            node = await self.store.get_paper_by_id(start_node_id)
            if not node:
                raise ValueError(f"Paper with ID '{start_node_id}' not found in database.")
            self.current_node = self._normalize_node_keys(node)
        else:
            # Search by title (query)
            # Note: Your store doesn't have search_papers_by_title, using get_paper_by_title instead
            candidates = await self.store.get_paper_by_title(query)
            
            if not candidates:
                raise ValueError(f"No starting node found for query: '{query}'")
            
            # Normalize the first candidate
            self.current_node = self._normalize_node_keys(candidates[0])

        # Verify we have a valid node
        if not self.current_node:
            raise ValueError(f"Failed to initialize current_node")
        
        # Mark initial node as visited
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
        # Get node text - use multiple fallbacks
        title = self.current_node.get('title', '')
        name = self.current_node.get('name', '')
        abstract = self.current_node.get('abstract', '')
        doi = self.current_node.get('doi', '')
        original_id = self.current_node.get('original_id', '')
        
        # Build text representation with all available info
        node_text = f"{title} {name} {abstract} {doi} {original_id}".strip()
        
        if not node_text or node_text == '':
            # Last resort: use paper_id itself (which is the elementId)
            node_text = self.current_node.get('paper_id', 'unknown_node')
        
        node_emb = self.encoder.encode(node_text if node_text else "empty")
    
        intent_vec = self._get_intent_vector(self.current_intent)
        
        # Concatenate: [Query(384) | Node(384) | Intent(5)] = 773
        return np.concatenate([self.query_embedding, node_emb, intent_vec])

    async def get_manager_actions(self) -> List[int]:
        """
        Manager decides which relation type to explore.
        Returns list of valid RelationType values.
        """
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        valid_relations = []
        
        # Determine node type and check available actions
        if paper_id:
            # Paper node - check paper-specific actions
            
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
            
            # SECOND_COLLAB (2nd degree collaborators)
            second_collabs = await self.store.get_second_degree_collaborators(author_id)
            if second_collabs:
                valid_relations.append(RelationType.SECOND_COLLAB)
            
            # INFLUENCE_PATH
            influence = await self.store.get_influence_path_papers(author_id)
            if influence:
                valid_relations.append(RelationType.INFLUENCE_PATH)
        
        # Always allow STOP
        valid_relations.append(RelationType.STOP)
        
        return valid_relations

    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        """
        Manager chooses a relation type.
        Returns: (is_terminal, manager_reward)
        """
        self.pending_manager_action = relation_type
        
        # If STOP chosen, episode ends
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps  # Force termination
            return True, 0.0
        
        # Manager reward: alignment with user intent
        if relation_type == self.current_intent:
            manager_reward = 1.0
        else:
            manager_reward = -0.3
        
        # Fetch available nodes for this relation type
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        raw_nodes = []
        
        # Execute the appropriate store function based on relation type
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
        
        # Normalize all node keys
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]
        
        # Create action tuples
        self.available_worker_nodes = [(node, relation_type) for node in normalized_nodes]
        
        # Filter out visited nodes
        self.available_worker_nodes = [
            (node, r_type) for node, r_type in self.available_worker_nodes 
            if not self._is_visited(node)
        ]
        
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
        
        # Ensure chosen_node is normalized (should already be, but just in case)
        self.current_node = self._normalize_node_keys(chosen_node)
        
        # Mark as visited
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        if paper_id:
            self.visited.add(paper_id)
        if author_id:
            self.visited.add(author_id)
        
        # Worker reward: semantic similarity to query
        title = self.current_node.get('title', '')
        name = self.current_node.get('name', '')
        abstract = self.current_node.get('abstract', '')
        doi = self.current_node.get('doi', '')
        original_id = self.current_node.get('original_id', '')
        
        node_text = f"{title} {name} {abstract} {doi} {original_id}".strip()
        
        if not node_text or node_text == '':
            node_text = self.current_node.get('paper_id', 'unknown_node')
        
        node_emb = self.encoder.encode(node_text if node_text else "empty")
        
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
        
        worker_reward = float(sem_reward)  
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        next_state = await self._get_state()
        
        # Clear pending action
        self.pending_manager_action = None
        self.available_worker_nodes = []
        
        return next_state, worker_reward, done

    # Keep legacy methods for backward compatibility (not used in hierarchical mode)
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