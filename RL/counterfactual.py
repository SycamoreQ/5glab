from typing import Optional , Tuple 
from RL.env import AdvancedGraphTraversalEnv
from RL.ddqn import DDQLAgent
from graph.database.store import EnhancedStore
from prioritized_replay import PrioritizedReplay


class TDError: 
    def __init__(self): 
        self.store = EnhancedStore(pool_size= 20 )
        self.env = AdvancedGraphTraversalEnv(store = self.store)
        
        self.replay_buff = PrioritizedReplay()


    def calculate_td_error(self , discount: float = 0.1) -> 