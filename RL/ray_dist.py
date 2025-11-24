import ray 
import torch 
import asyncio
import logging
import time 
import os 
from collections import deque
import numpy as np 
from env import AdvancedGraphTraversalEnv, RelationType
from ddqn import DDQLAgent
from graph.database.store import EnhancedStore
import random 


@ray.remote
class SharedStorage:
    def __init__(self):
        self.memory = deque(maxlen = 500000)
        self.weights = None 
        self.target_weights = None 

    def add_experience(self, experience):
        self.memory.append(experience)
    
    def sample_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]
    
    def set_weights(self, weights):
        self.weights = weights
    
    def get_weights(self):
        return self.weights

    def size(self):
        return len(self.memory)
    

@ray.remote
class Explorer:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        # Connect to the DB on Machine A
        # Note: If running ON Machine A, use localhost. If elsewhere, use IP.
        # We assume Memgraph accepts remote connections or this worker is forced to Machine A.
        self.store = EnhancedStore() 
        self.env = AdvancedGraphTraversalEnv(self.store)
        self.agent = DDQLAgent(773, 384)
        self.agent.epsilon = 0.4 + (worker_id * 0.05)

    async def run_exploration(self , storage_actor):
        logging.info(f"Learner {self.worker_id} started")
        
        scenarios = [
            "deep learning transformers", RelationType.CITES,
            "graph neural networks", RelationType.CITED_BY,
            "yann lecun", RelationType.WROTE,
            "reinforcement learning", RelationType.CITES
        ]

        while True:
            if random.random() < 0.1:
                weights = await storage_actor.get_weights.remote()
                
                if weights: 
                    self.agent.policy_net.load_state_dict(weights)
                    logging.info(f"Learner {self.worker_id} updated weights from storage.")

                    
                query , intent = random.choice(scenarios)
                
                try:
                    state = await self.env.reset(query, intent)
                except Exception:
                    continue

                done = False
                while not done:
                    actions = await self.env.get_valid_actions()
                    if not actions: break
                    
                    action_tuple = self.agent.act(state, actions) 
                    if not action_tuple: break
                    
                    node, relation = action_tuple
                    next_state, reward, done = await self.env.step(node, relation)
                    next_actions = await self.env.get_valid_actions()

                    # 3. Push to Storage
                    # We store minimal data to reduce network overhead
                    experience = (state, action_tuple, reward, next_state, done, next_actions)
                    storage_actor.add_experience.remote(experience)
                
                    state = next_state
                


@ray.remote(num_gpus=0.5)
class Learner:
    def __init__(self , storage_actor):
        self.agent = DDQLAgent(773, 384)
        self.storage = storage_actor 
        self.steps = 0 

    
    async def run_learning(self):
        