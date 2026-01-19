
import ray
import time
import random
from typing import Dict


@ray.remote
class MockRLTrainer:
    def __init__(self, worker_id: str, **kwargs):
        self.worker_id = worker_id
        print(f"[MOCK] MockRLTrainer {worker_id} initialized")
    
    async def train_episodes(
        self,
        episodes: int,
        query: str,
        start_paper_id: str,
        **kwargs
    ) -> Dict:
        """
        Simulate training by sleeping and returning fake results.
        """
        print(f"[MOCK] {self.worker_id} starting mock training")
        print(f"  Query: {query}")
        print(f"  Episodes: {episodes}")
        print(f"  Start paper: {start_paper_id}")
        
        training_time = episodes * 0.5
        
        print(f"[MOCK] Simulating {training_time:.1f}s of training...")
        time.sleep(min(training_time, 10)) 
        
        result = {
            'job_type': 'mock_rl_training',
            'worker_id': self.worker_id,
            'episodes': episodes,
            'duration_sec': training_time,
            'avg_reward': random.uniform(10, 50),
            'max_reward': random.uniform(50, 100),
            'final_reward': random.uniform(20, 60),
            'avg_similarity': random.uniform(0.3, 0.8),
            'max_similarity': random.uniform(0.6, 0.9),
            'final_similarity': random.uniform(0.4, 0.7),
            'avg_loss': random.uniform(0.1, 1.0),
            'final_epsilon': random.uniform(0.1, 0.3),
            'checkpoint_path': f'mock_checkpoints/{self.worker_id}_mock.pt',
            'status': 'completed'
        }
        
        print(f"[MOCK] {self.worker_id} training complete!")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")
        print(f"  Max Similarity: {result['max_similarity']:.3f}")
        
        return result


# Alias for compatibility
DistributedRLTrainer = MockRLTrainer