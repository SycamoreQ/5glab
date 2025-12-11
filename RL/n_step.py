from collections import deque
from typing import Optional, Tuple
import numpy as np


class NStepBuffer:
    def __init__(self, n=3, gamma=0.95):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)
    
    def add(self, state, action_emb, r_type, reward, next_state, done, next_actions):
        self.buffer.append((state, action_emb, r_type, reward, next_state, done, next_actions))
        
        if len(self.buffer) >= self.n or done:
            return self.compute_n_step_return()
        
        return None
    
    def compute_n_step_return(self):
        if not self.buffer:
            return None
        
        state, action_emb, r_type, _, _, _, _ = self.buffer[0]
        
        n_step_reward = 0
        for i, (_, _, _, reward, _, done, _) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                break
        
        _, _, _, _, next_state, done, next_actions = self.buffer[-1]
        
        result = (state, action_emb, r_type, n_step_reward, next_state, done, next_actions)
        self.buffer.popleft()
        
        return result
    
    def flush(self):
        transitions = []
        while self.buffer:
            trans = self.compute_n_step_return()
            if trans:
                transitions.append(trans)
        return transitions
