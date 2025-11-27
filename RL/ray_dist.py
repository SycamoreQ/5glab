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
    
    def get_batch(self , batch_size):
        if len(self.memory) < batch_size:
            return []
        
        random.sample(self.memory , batch_size)


    def size(self):
        return len(self.memory)
    

@ray.remote
class Explorer:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.store = EnhancedStore() 
        self.env = AdvancedGraphTraversalEnv(self.store)
        self.agent = DDQLAgent(773, 384)
        self.agent.epsilon = 0.4 + (worker_id * 0.05)

    async def run_episode(self , episode_idx:int): 
        weights = ray.get(self)

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

    
def update_model(self):
        self.storage.set_weights.remote(
            {k: v.cpu() for k, v in self.agent.policy_net.state_dict().items()}
        )

        while True:
            batch = ray.get(self.storage.get_batch.remote(32))
            
            if not batch:
                time.sleep(1)
                continue

            self.agent.memory = batch
            self.agent.replay()

            self.steps += 1

            if self.steps % 50 == 0 : 
                self.storage.set_weights.remote(
                    {self.k : v.cpu() for k , v in self.agent.policy.net_state_dict().items()}
                )

                print(f"Learner Step {self.steps} | Loss optimized")

            if self.steps % 50 == 0: 
                self.agent.update_target_network()
                torch.save(
                    self.agent.policy_net.state_dict(),
                    f"models/ddqn_learner_checkpoint.pth"
                )

