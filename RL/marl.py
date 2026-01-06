import asyncio
import numpy as np
import torch
from collections import deque
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv
from graph.database.store import EnhancedStore


@dataclass
class AgentExperience: 
    agent_id: int
    episode: int
    reward: float
    steps: int
    communities_visited: int
    similarity: float
    query: str
    trajectory: List[Dict]



class SharedExperienceBuffer:
    
    def __init__(self , max_size: int = 500000): 
        self.buffer = deque(maxlen=max_size)
        self.high_quality_buffer = deque(maxlen=10000)
        self.quality_threshold = 10.0

    
    def add(self , experience: AgentExperience): 
        self.buffer.append(experience)

        if experience.reward >= self.quality_threshold: 
            self.high_quality_buffer.append(experience)


    def sample(self , batch_size: int , prioritize_quality: float = 0.3): 
        if len(self.buffer) < batch_size:
            return []
        
        high_quality_samples = int(batch_size * prioritize_quality)
        regular_samples = batch_size - high_quality_samples
        
        samples = []
        
        if len(self.high_quality_buffer) > 0:
            high_quality_samples = min(high_quality_samples, len(self.high_quality_buffer))
            samples.extend(np.random.choice(
                list(self.high_quality_buffer),
                size=high_quality_samples,
                replace=False
            ))
        
        if len(self.buffer) >= regular_samples:
            samples.extend(np.random.choice(
                list(self.buffer),
                size=regular_samples,
                replace=False
            ))
        
        return samples

    
    def get_best_trajectory(self , k: int = 10) -> List[AgentExperience]: 
        if len(self.buffer) == 0: 
            return []
        
        sorted_experience = sorted(
            self.buffer ,
            key = lambda x : x.reward , 
            reverse=True
        )

        return sorted_experience[:k]
    
    

class AgentWorker: 
    def __init__(self , agent_id: int , store: EnhancedStore , state_dim: int = 783 , text_dim: int = 384 , use_communities: bool = True): 
        self.agent_id = agent_id
        self.agent = DDQLAgent(state_dim=state_dim, text_dim=text_dim)
        self.env = AdvancedGraphTraversalEnv(store, use_communities=use_communities)
        self.episode_count = 0


    async def run_episode(self , query:str , start_paper: Dict , intent: int = 1) -> AgentExperience: 
        try: 
            state = await self.env.reset(
                query, 
                intent, 
                start_node_id= start_paper['paper_id']
            )

            episode_reward = 0.0 
            trajectory = []
            
            for step in range(self.env.max_steps) : 
                manager_actions = await self.env.get_manager_actions()

                if not manager_actions: 
                    break 

                manager_action = 1 if 1 in manager_actions else np.random.choice(manager_actions)
                is_terminal, manager_reward = await self.env.manager_step(manager_action)
                episode_reward += manager_reward


                if is_terminal: 
                    break 

                worker_action = await self.env.get_worker_actions()

                if not worker_action:
                    break 

                best_action = self.agent.act(state , worker_action)
                if not best_action: 
                    break 

                chosen_node , _ = best_action

                next_state , worker_reward , done  = await self.env.worker_step(chosen_node)
                episode_reward += worker_reward 

                if not done:
                    next_manager_action = await self.env.get_manager_actions()
                    if next_manager_action: 
                        temp_manager = 1 if 1 in next_manager_action else next_manager_action[0]
                        await self.env.manager_step(temp_manager)
                        next_worker_actions = await self.env.get_worker_actions()
                        self.env.pending_manager_action = None
                        self.env.available_worker_nodes = []
                    else: 
                        next_worker_actions = []

                else:
                    next_worker_actions = []

                self.agent.remember(
                    state=state,
                    action_tuple=best_action,
                    reward=episode_reward,
                    next_state=next_state,
                    done=done,
                    next_actions=next_worker_actions
                )
                
                trajectory.append({
                    'step': step,
                    'node': chosen_node,
                    'reward': worker_reward,
                    'similarity': self.env.best_similarity_so_far
                })
                
                state = next_state
                
                if done:
                    break

            summary = self.env.get_episode_summary()
            self.episode_count += 1      
            
            
            return AgentExperience(
                agent_id=self.agent_id,
                episode=self.episode_count,
                reward=episode_reward,
                steps=summary['path_length'],
                communities_visited=summary['unique_communities_visited'],
                similarity=summary['max_similarity_achieved'],
                query=query,
                trajectory=trajectory
            )
        
        except Exception as e: 
            logging.error(f"Agent {self.agent_id} episode failed: {e}")
            return AgentExperience(
                agent_id=self.agent_id,
                episode=self.episode_count,
                reward=0.0,
                steps=0,
                communities_visited=0,
                similarity=0.0,
                query=query,
                trajectory=[]
            )
            

    def train_on_batch(self , batch_size: int = 32):
        if len(self.agent.memory) >= batch_size: 
            import random
            batch = random.sample(list(self.agent.memory), batch_size)
            loss = self.agent.replay_batch(batch)
            self.agent.epsilon = max(0.1, self.agent.epsilon * 0.995)
            return loss
        return 0.0
    
    def sync_from_best(self, best_agent_state: Dict):
        self.agent.policy_net.load_state_dict(best_agent_state)
        
                    


class MultiAgentTrainer: 

    def __init__(self , store = EnhancedStore() , num_agents: int = 4 , use_communities:bool = True): 
        self.store = store 
        self.num_agents = num_agents
        self.agents = [
            AgentWorker(i , store , use_communities= use_communities) for i in range(num_agents)
            
        ]

        self.shared_buffer = SharedExperienceBuffer()
        self.global_eps = 0 

    async def train_parallel_episode(self, queries: List[str] , start_papers: List[Dict]) -> List[AgentExperience]: 
        tasks = []

        for i , agent in enumerate(self.agents): 
            query_idx = i % len(queries)
            paper_idx = i % len(start_papers)
            
            task = agent.run_episode(query = queries[query_idx],
                                     start_paper= start_papers[paper_idx])
            tasks.append(task)


        experiences = await asyncio.gather(*tasks)
        
        for exp in experiences:
            self.shared_buffer.add(exp)
            
        
        return experiences
    
    
    def train_all_agents(self, batch_size: int = 32):
        losses = []
        
        for agent in self.agents:
            loss = agent.train_on_batch(batch_size)
            losses.append(loss)
        
        return np.mean(losses) if losses else 0.0


    def sync(self):
        best = max(self.agents, 
               key=lambda x: sum(exp.reward for exp in self.shared_buffer.buffer 
                                 if exp.agent_id == x.agent_id))

        best_state = best.agent.policy_net.state_dict()

        for agent in self.agents: 
            if agent.agent_id != best.agent_id: 
                agent.agent.policy_net.load_state_dict(best_state)

        
    def update_target_network(self): 
        for agent in self.agents: 
            agent.agent.update_target()

    def get_statistics(self) -> Dict[str , Any]:

        best_trajectories = self.shared_buffer.get_best_trajectory(k = 5)

        return {
            'total_experiences': len(self.shared_buffer.buffer),
            'high_quality_experiences': len(self.shared_buffer.high_quality_buffer),
            'best_reward': max([exp.reward for exp in best_trajectories]) if best_trajectories else 0.0,
            'avg_reward_top5': np.mean([exp.reward for exp in best_trajectories]) if best_trajectories else 0.0,
            'agent_episodes': [agent.episode_count for agent in self.agents]
        } 

            
        
        


            
    