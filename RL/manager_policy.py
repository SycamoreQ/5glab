import numpy as np 
import torch 
import torch.nn as nn
from enum import Enum
from typing import List, Tuple, Dict
from RL.env import RelationType


class ManagerStrategy(Enum): 
    EXPLOIT = 0
    EXPLORE = 1
    BRIDGE = 2
    DEEP_DIVE = 3
    TEMPORAL = 4


class ManagerPolicyNetwork(nn.Module): 
    
    def __init__(self, state_dim: int, num_strats: int = 5): 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Linear(128, num_strats)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor: 
        return self.net(state)
    

class AdaptiveManagerPolicy:
    
    def __init__(self, state_dim: int = 773): 
        self.state_dim = state_dim
        self.policy_net = ManagerPolicyNetwork(state_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        self.strategy_rewards = {s: [] for s in ManagerStrategy} 
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def select_strategy( 
        self,
        state: np.ndarray,
        episode_progress: float,
        visited_communities: int,
        current_reward: float) -> ManagerStrategy: 
        
        if np.random.random() < self.epsilon: 
            return np.random.choice(list(ManagerStrategy))  
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        strategy_idx = q_values.argmax().item()
        return ManagerStrategy(strategy_idx) 

    def get_relation_for_strategy(  
        self, 
        strategy: ManagerStrategy,  
        available_relations: List[int], 
        env_state: Dict
    ) -> int: 
        
        if strategy == ManagerStrategy.EXPLOIT:  
            if RelationType.CITED_BY in available_relations: 
                return RelationType.CITED_BY
            return available_relations[0]
        
        elif strategy == ManagerStrategy.EXPLORE:  
            exploration_types = [
                RelationType.KEYWORD_JUMP,
                RelationType.VENUE_JUMP,
                RelationType.COLLAB
            ]
            for rel in exploration_types: 
                if rel in available_relations: 
                    return rel 
            return available_relations[0] 
            
        elif strategy == ManagerStrategy.BRIDGE:
            bridge_types = [
                RelationType.SECOND_COLLAB,
                RelationType.INFLUENCE_PATH
            ]
            for rel in bridge_types:
                if rel in available_relations:
                    return rel
            return RelationType.CITES if RelationType.CITES in available_relations else available_relations[0] 
            
        elif strategy == ManagerStrategy.DEEP_DIVE:  
            if RelationType.OLDER_REF in available_relations: 
                return RelationType.OLDER_REF
            elif RelationType.CITES in available_relations:
                return RelationType.CITES
            return available_relations[0]
        
        elif strategy == ManagerStrategy.TEMPORAL:  
            if RelationType.NEWER_CITED_BY in available_relations:
                return RelationType.NEWER_CITED_BY
            return RelationType.CITED_BY if RelationType.CITED_BY in available_relations else available_relations[0]
        
        return available_relations[0]

    def update_policy(self, state: np.ndarray, strategy: ManagerStrategy, reward: float):  
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        q_values = self.policy_net(state_tensor)
        target = q_values.clone()
        target[0, strategy.value] = reward

        loss = nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.strategy_rewards[strategy].append(reward)
        
    def get_strategy_stats(self) -> Dict:
        stats = {}
        for strategy, rewards in self.strategy_rewards.items():
            if rewards:
                stats[strategy.name] = {
                    'count': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'best_reward': max(rewards)
                }
            else:
                stats[strategy.name] = {'count': 0, 'avg_reward': 0.0, 'best_reward': 0.0}
        return stats
