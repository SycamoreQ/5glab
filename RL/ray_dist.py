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
from typing import List, Tuple, Dict, Any

@ray.remote(resources= {"shared_storage": 1})
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
    
    def log_reward(self , reward):
        print(f"Episode Reward: {reward}")
        
    def get_batch(self , batch_size):
        if len(self.memory) < batch_size:
            return []
        
        return random.sample(self.memory , batch_size)

    def size(self):
        return len(self.memory)
    

@ray.remote(num_cpus=1 , resources = {"worker_node": 0.1}) 
class Explorer:
    def __init__(self, worker_id , storage_actor):
        self.worker_id = worker_id
        self.store = EnhancedStore() 
        self.env = AdvancedGraphTraversalEnv(self.store)
        self.agent = DDQLAgent(773, 384)
        self.storage = storage_actor

    
    def _select_action_with_beam_search(self, state: np.ndarray, worker_action: List[Tuple[Dict, RelationType]]) -> Optional[Tuple[Dict, RelationType]]:
        """
        Performs the greedy action selection (equivalent to a one-step beam search 
        where beam width equals the number of available actions).
        """
        best_q_value = -float('inf')
        best_action_tuple = None # (node_dict, rel_type)

        for node_dict, rel_type in worker_action:
            node_text = node_dict.get('title') or node_dict.get('name') or ''
            node_emb = self.agent.encoder.encode(node_text)
            
            rel_onehot = self.agent._get_relation_onehot(rel_type).squeeze(0).cpu().numpy()
            
            q_value = self.agent.predict_q_value(state, node_emb, rel_onehot)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action_tuple = (node_dict, rel_type)

        return best_action_tuple

    async def _process_hindsight_experience(self, trajectory: List[Tuple]):
        """
        Generates Hindsight Experience Replay (HER) transitions using the Final Goal strategy.
        """
        if not trajectory:
            return

        final_node = self.env.current_node 
        final_node_txt = final_node.get('title') or final_node.get('name') or ' '
        hindsight_query_emb = self.agent.encoder.encode(final_node_txt) # New goal embedding
        
        for i, (state, action_tuple, original_reward, next_state, done, worker_actions) in enumerate(trajectory): 
            
            new_state = state.copy()
            new_state[:self.agent.text_dim] = hindsight_query_emb
            
            new_next_state = next_state.copy()
            new_next_state[:self.agent.text_dim] = hindsight_query_emb
            
            node_emb = new_state[self.agent.text_dim:self.agent.text_dim * 2]
            
            new_sem_reward = np.dot(hindsight_query_emb, node_emb) / (
                np.linalg.norm(hindsight_query_emb) * np.linalg.norm(node_emb) + 1e-9
            )
            
            new_total_reward = new_sem_reward 
            
            if i == len(trajectory) - 1:
                new_total_reward += 5.0 
            
            h_exp = (new_state, action_tuple, new_total_reward, new_next_state, done, worker_actions)
            
            self.storage.add_experience.remote(h_exp)

    async def run_episode(self , episode_idx:int): 
        print(f"Episode {episode_idx} started")
        
        weights = ray.get(self.storage.get_weights.remote())
        
        if weights: 
            self.agent.policy_net.load_state_dict(weights)
            self.agent.epsilon = max(0.1, 0.999**episode_idx) 

        initial_query = "papers on large language models and reinforcement learning" 
        initial_intent = RelationType.CITED_BY 
        state = await self.env.reset(initial_query, initial_intent)

        done = False 
        total_reward = 0.0
        local_memory = []
        
        while not done: 
            valid_manager_actions = self.env.get_manager_actions()
            manager_action_reltype = random.choice(valid_manager_actions)

            is_terminal , manager_reward = self.env.manager_step(manager_action_reltype)
            
            if is_terminal: 
                action_emb = np.zeros(self.agent.text_dim)
                action_tuple = (action_emb, manager_action_reltype)
                experience = (state, action_tuple, manager_reward, state, True, [])
                local_memory.append(experience) 
                done = True 
                break

            worker_action = self.env.get_worker_actions()

            if not worker_action:
                state = await self.env._get_state()
                continue 

            if random.random() <  self.agent.epsilon:
                chosen_node_dict, chosen_rel_type = random.choice(worker_action)
            else:
                best_action_tuple = self._select_action_with_beam_search(state, worker_action)
                
                if best_action_tuple:
                    chosen_node_dict, chosen_rel_type = best_action_tuple
                else:
                    chosen_node_dict, chosen_rel_type = random.choice(worker_action)

            
            action_emb = self.agent.encoder.encode(chosen_node_dict.get('title') or chosen_node_dict.get('name') or "")
            action_tuple = (action_emb, manager_action_reltype) 

            next_state, worker_reward, done = await self.env.worker_step(chosen_node_dict)
            
            current_total_reward = manager_reward + worker_reward
            total_reward += current_total_reward

            experience = (state, action_tuple, current_total_reward, next_state, done, worker_action)
            local_memory.append(experience)
            
            state = next_state

        
        for exp in local_memory: 
            self.storage.add_experience.remote(exp)
            
        if len(local_memory) > 0: 
            await self._process_hindsight_experience(local_memory)
                
        self.storage.log_reward.remote(total_reward)

        return episode_idx, total_reward , local_memory


@ray.remote(num_gpus=1 , resources={"learner_node" : 1})
class Learner:
    def __init__(self , storage_actor):
        self.agent = DDQLAgent(773, 384)
        self.storage = storage_actor 
        self.steps = 0 
        self.batch_size = 32 

    def update_model(self):
        self.storage.set_weights.remote(
            {k: v.cpu() for k, v in self.agent.policy_net.state_dict().items()}
        )

        while True:
            batch = ray.get(self.storage.get_batch.remote(self.batch_size))
            
            if not batch:
                time.sleep(1)
                continue

            loss = self.agent.replay_batch(batch) 

            self.steps += 1

            if self.steps % 50 == 0 : 
                self.storage.set_weights.remote(
                    {k : v.cpu() for k , v in self.agent.policy_net.state_dict().items()}
                )
                print(f"Learner Step {self.steps} | Loss: {loss:.4f} | Storage Size: {ray.get(self.storage.size.remote())}")

            if self.steps % 50 == 0: 
                self.agent.update_target()
                torch.save(
                    self.agent.policy_net.state_dict(),
                    f"models/ddqn_learner_checkpoint.pth"
                )