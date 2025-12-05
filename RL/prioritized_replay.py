import numpy as np
from collections import namedtuple
from typing import List, Tuple

Experience = namedtuple('Experience', ['state', 'action', 'r_type', 'reward', 'next_state', 'done', 'next_actions'])


class PrioritizedReplay:
    """Prioritized Experience Replay Buffer."""
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, r_type, reward, next_state, done, next_actions):
        """Add experience to buffer with relation type."""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        experience = Experience(state, action, r_type, reward, next_state, done, next_actions)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with priorities."""
        if self.size < batch_size:
            return [], np.array([]), np.array([])
        
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        """Return current buffer size."""
        return self.size
    
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
