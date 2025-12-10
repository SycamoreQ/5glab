import numpy as np
from collections import deque
from typing import List, Tuple, Any

class PrioritizedReplay:
    def __init__(self, max_size=100000, alpha=0.6, beta=0.4 , capacity = None):
        if capacity is not None:
            max_size = capacity 
        self.max_size = max_size   
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        self.size = 0
    
    def add(self, state, action, relation_type, reward, next_state, done, next_actions):
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0
        
        self.buffer.append((state, action, relation_type, reward, next_state, done, next_actions))
        self.priorities.append(max_priority ** self.alpha)
        self.size = len(self.buffer)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        priorities_array = np.array(self.priorities)
        probs = priorities_array / priorities_array.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        """Make buffer iterable for compatibility."""
        if self.size == 0:
            return iter([])
        return iter(self.buffer[:self.size])
    
    def __getitem__(self, idx):
        """Allow indexing."""
        if idx < self.size:
            return self.buffer[idx]
        raise IndexError(f"Index {idx} out of range for buffer size {self.size}")
    
    def get_stats(self):
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'avg_priority': 0.0,
                'max_priority': 0.0,
                'min_priority': 0.0,
                'beta': self.beta
            }
        
        priorities = self.priorities[:self.size]
        return {
            'size': self.size,
            'capacity': self.capacity,
            'avg_priority': float(priorities.mean()),
            'max_priority': float(priorities.max()),
            'min_priority': float(priorities.min()),
            'beta': self.beta
        }
