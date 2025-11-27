import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from graph.database.store import * 
from sentence_transformers import SentenceTransformer
from action_dispatch import ACTION_DISPATCH, ACTION_ARG_MAP , RelationType
import logging


class AdvancedGraphTraversalEnv:
    """
    Goal-Conditioned Environment.
    The 'Goal' is defined by the User's Intent (e.g., "Find References").
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

    def _get_intent_vector(self, intent_enum: int):
        vec = np.zeros(self.intent_dim)
        vec[intent_enum] = 1.0
        return vec

    async def reset(self, query: str, intent: int, start_node_id: str = None):
        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
        self.current_step = 0
        
        if not start_node_id:
            candidates = await self.store.search_papers_by_title(query)
            if not candidates:
                raise ValueError("No starting node found.")
            self.current_node = candidates[0]
        else:
            self.current_node = await self.store.get_paper_by_id(start_node_id)

        self.visited.add(self.current_node['paper_id'])
        return self._get_state()

    def _get_state(self):
        node_text = f"{self.current_node.get('title', '')} {self.current_node.get('abstract', '')}"
        node_emb = self.encoder.encode(node_text)
    
        intent_vec = self._get_intent_vector(self.current_intent)
    
        return np.concatenate([self.query_embedding, node_emb, intent_vec])

    async def get_valid_actions(self) -> List[Tuple[Dict, int]]:
        node = self.current_node
        node_type = node.get("type", "Paper")  # Default to "Paper" if not set
        actions = []
        
        for (rel_type, src_type), store_func_name in ACTION_DISPATCH.items():
            if src_type != node_type:
                continue
            func = getattr(self.store, store_func_name)
            arg_key = ACTION_ARG_MAP[(rel_type, src_type)]
            arg_val = node.get(arg_key)
            if not arg_val:
                continue
            # Call the store function; must be awaited as it's async
            results = await func(arg_val)
            for res in results:
                res["type"] = "Paper" if rel_type in {RelationType.CITES, RelationType.CITED_BY, RelationType.AUTHORED} else "Author"
                actions.append((res, rel_type))
        
        valid_actions = [
            (node, r_type) for node, r_type in actions
            if node.get('paper_id') not in self.visited and node.get('author_id') not in self.visited
        ]

        return valid_actions

    async def step(self, action_node: Dict, action_relation: int):
        self.current_step += 1
        self.current_node = action_node
        identifier = action_node.get('paper_id') or action_node.get('author_id')
        self.visited.add(identifier)
        if action_relation == self.current_intent:
            struct_reward = 1.0
        else:
            struct_reward = -0.5  # Soft penalty (sometimes accidental discovery is okay, but discouraged)
        node_text = action_node.get('title', '') or action_node.get('name', '')
        node_emb = self.encoder.encode(node_text)
        
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )

        total_reward = (0.6 * struct_reward) + (0.4 * sem_reward)
        
        done = self.current_step >= self.max_steps
        return self._get_state(), total_reward, done