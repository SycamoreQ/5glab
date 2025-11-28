import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from .ddqn import DDQLAgent
from graph.database.store import EnhancedStore
from .action_dispatch import ACTION_SPACE_MAP, ACTION_VALID_SOURCE_TYPE, RelationType

class AdvancedGraphTraversalEnv:
    """
    Hierarchical Graph Traversal Environment.
    The Agent chooses a Relation (Manager), then a Node (Worker).
    """ 
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2"):
        self.store = store
        self.encoder = SentenceTransformer(embedding_model_name)
        self.query_embedding = None
        self.store = EnhancedStore()
        self.current_intent = None 
        self.current_node = None
        self.visited = set()
        self.max_steps = 5
        self.current_step = 0
        self.text_dim = 384 
        self.intent_dim = 13 
        self.state_dim = self.text_dim * 2 + self.intent_dim
        self.manager_action_space = list(ACTION_SPACE_MAP.keys())
        self.current_manager_action = None
        self.node_quality_cache: Dict[str , float] = {}

    def _get_intent_vector(self, intent_enum: int):
        vec = np.zeros(self.intent_dim)
        if 0 <= intent_enum < self.intent_dim:
            vec[intent_enum] = 1.0
        return vec
    
    def _normalize(value, max_val, min_val=0):
        """Min-max normalization."""
        if max_val <= min_val: return 0.0
        return (value - min_val) / (max_val - min_val)
    
    async def _get_node_quality(self , node: Dict ) -> float: 
        node_type = self._determine_node_type
        node_id = node.get("paper_id") or node.get("author_id")

        if node_id in self.node_quality_cache:
            return self.node_quality_cache[node_id]


        MAX_CITATION_COUNT = 100 
        MAX_COLLAB_COUNT = 50
        score = 0.0 

        if node_type == "Paper": 
            paper_id = node.get("paper_id")
            citation_count = self.store.get_citation_count(paper_id)
            citation_count = citation_count[0]['count']

            score = np.log(citation_count)/np.log(MAX_CITATION_COUNT)
            return score 
        
        elif node_type == "Author":
            author_id = node.get("author_id")
            collab_count = self.store.get_collab_count(author_id)
            collab_count = collab_count[0]['count']

            score = np.log1p(collab_count) / np.log1p(MAX_COLLAB_COUNT)
            return score
        

        self.node_quality_cache[node_id] = score
        return score 
    
    
    def _determine_node_type(self, node: Dict) -> str:
        """Helper to classify current node."""
        if "paper_id" in node: return "Paper"
        if "author_id" in node: return "Author"
        return "Unknown"

    async def reset(self, query: str, intent: int, start_node_id: str = None):
        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
        self.current_step = 0
        self.current_manager_action = None
        self.node_quality_cache = {}

        
        if not start_node_id:
            candidates = await self.store.get_paper_by_title(query)
            if not candidates: raise ValueError("No starting node found.")
            self.current_node = candidates[0]
        else:
            self.current_node = await self.store.get_paper_by_id(start_node_id)
            if not self.current_node:
                 raise ValueError(f"Start node ID {start_node_id} not found in DB.")

        identifier = self.current_node.get('paper_id') or self.current_node.get('author_id')
        self.visited.add(identifier)
        return self._get_state()

    async def _get_state(self):
        """Current state is always (Query_Emb, Node_Emb, Intent_Vec)."""
        text_content = self.current_node.get('title') or self.current_node.get('name') or ""
        abstract = self.current_node.get('abstract', "")
        
        node_text = f"{text_content} {abstract}"
        node_emb = self.encoder.encode(node_text)    
        intent_vec = self._get_intent_vector(self.current_intent)

        impact_score = await self._get_node_quality(self.current_node)
        impact_vec = np.array([impact_score])

    
        return np.concatenate([self.query_embedding, node_emb, intent_vec , impact_vec])


    async def get_manager_actions(self) -> List[int]:
        current_type = self._determine_node_type(self.current_node)
        
        valid_actions = [RelationType.STOP] 
        
        for rel_type, src_type in ACTION_VALID_SOURCE_TYPE.items():
            if rel_type == RelationType.STOP:
                continue
            
            if src_type == current_type:
                valid_actions.append(rel_type)
        
        return valid_actions



    async def _resolve_store_call(self, func_name: str, rel_type: int, node: Dict):
        func = getattr(self.store, func_name, None)
        if not func: return []
        
        if "paper_id" in func_name and "paper_id" in node:
            return await func(node["paper_id"])
        if "author_id" in func_name and "author_id" in node:
            return await func(node["author_id"])
            
        if rel_type == RelationType.KEYWORD_JUMP:
            keywords = node.get("keywords", "")
            if not keywords: return []
            kw = keywords.split(',')[0].strip() if isinstance(keywords, str) else keywords[0].strip()
            return await func(kw, limit=5, exclude_paper_id=node.get("paper_id"))

        if rel_type == RelationType.VENUE_JUMP:
            venue = node.get("publication_name") or node.get("venue")
            if not venue: return []
            return await func(venue, limit=5, exclude_paper_id=node.get("paper_id"))
            
        if rel_type == RelationType.INFLUENCE_PATH:
            return await func(node["author_id"], limit=5)

        return []

    async def get_worker_actions(self) -> List[Tuple[Dict, int]]:
        if self.current_manager_action is None:
            return []
            
        rel_type = self.current_manager_action
        func_name = ACTION_SPACE_MAP[rel_type]

        try:
            neighbors = await self._resolve_store_call(func_name, rel_type, self.current_node)
            
            possible_moves = []
            for neighbor in neighbors:
                neighbor_id = neighbor.get('paper_id') or neighbor.get('author_id')
                if neighbor_id and neighbor_id not in self.visited:
                    neighbor["type"] = self._determine_node_type(neighbor)
                    possible_moves.append((neighbor, rel_type))
            
            return possible_moves

        except Exception as e:
            logging.error(f"Error executing worker action {func_name}: {e}")
            return []

                

                    
    async def manager_step(self, action_relation: int) -> Tuple[bool, float]:
        self.current_manager_action = action_relation
        manager_reward = 0.0 
        
        if action_relation == RelationType.STOP:
            is_terminal = True
            
            current_text = self.current_node.get('title') or self.current_node.get('name') or ""
            current_emb = self.encoder.encode(current_text)
            sem_match = np.dot(self.query_embedding, current_emb) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(current_emb) + 1e-9
            )

            impact_score = await self._get_node_quality(self.current_node)
            time_efficiency = 1.0 - self._normalize(self.current_step, self.max_steps)

            terminal_reward = (4.0*sem_match) + (2.0*impact_score) * (2.0*time_efficiency)
            
            return is_terminal , terminal_reward
        
        manager_reward += 1.0 if action_relation == self.current_intent else -0.2
    
        neighbors = await self.get_worker_actions()
        if not neighbors:
                manager_reward -= 2.0 
         
        manager_reward -= 0.05 
             
        return False, manager_reward
    

    async def worker_step(self, action_node: Dict) -> Tuple[np.ndarray, float, bool]:
        self.current_step += 1
        self.current_node = action_node

        identifier = action_node.get('paper_id') or action_node.get('author_id')
        self.visited.add(identifier)

        text_content = action_node.get('title') or action_node.get('name') or ""
        node_emb = self.encoder.encode(text_content)
        sem_match = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
            
        node_quality = await self._get_node_quality(action_node)

        total_reward = (0.7 * sem_match) + (0.3 * node_quality)

        done = self.current_step >= self.max_steps

        self.current_manager_action = None 

        return self._get_state(), total_reward, done