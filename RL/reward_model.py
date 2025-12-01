import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from env import AdvancedGraphTraversalEnv , RelationType

class RewardModel:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)

    def _cosine_similarity(self, vec1, vec2):
        vec1 = torch.tensor(vec1)
        vec2 = torch.tensor(vec2)
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

    def compute_reward(self, query, action_node, intent_type):
        node_text = action_node.get('title', '') or action_node.get('name', '')
        if 'abstract' in action_node:
            node_text += " " + action_node['abstract']
        node_emb = self.encoder.encode(node_text)
        query_emb = self.encoder.encode(query)
        sim_reward = self._cosine_similarity(query_emb, node_emb)

        citation_count = action_node.get('citation_count', 0)
        citation_score = np.log(1 + citation_count) / 10.0

        year = action_node.get('year')
        recency = 0.0
        if year is not None:
            recency = np.exp(-0.1 * (2025 - year)) 

        author_impact = 0.0

        intent_bonus = 1.0 if action_node.get('relation') == intent_type else 0.0
        reward = (
            0.4 * sim_reward +
            0.2 * citation_score +
            0.1 * recency +
            0.1 * intent_bonus +
            0.1 * author_impact
        )
        return reward
    
    def compute_reward_2(self , query , action_node , action_relation):
        #calculate cosine sim 
        node_text = action_node.get('title', '') or action_node.get('name', '')
        if 'abstract' in action_node:
            node_text += " " + action_node['abstract']
        node_emb = self.encoder.encode(node_text)
        query_emb = self.encoder.encode(query)
        sim_reward = self._cosine_similarity(query_emb, node_emb)
        

        #calculate graph structural similarity
        #2 hop reward , k-hop reward and so on..

        if action_relation is RelationType.CITES or RelationType.CITED_BY : 
            if 

        

        

