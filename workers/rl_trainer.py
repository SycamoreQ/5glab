import ray
import time
import torch
import numpy as np
import asyncio
from workers.base_worker import BaseWorker
from workers.remote_store import RemoteStore
from typing import Dict , List , Optional


@ray.remote(num_gpus=3, num_cpus=0)
class DistributedRLTrainer(BaseWorker):
    """
    Distributed RL trainer for parallel training.
    Each worker trains independently on remote database.
    """
    
    def __init__(
        self,
        worker_id: str,
        db_host: str,
        db_port: int = 7687,
        db_user: str = "neo4j",
        db_password: str = "diam0ndman@3",
        db_name: str = "researchdbv3"
    ):
        super().__init__(worker_id, node_type="rl_trainer")
        
        self.store = RemoteStore(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            pool_size=5
        )
        
        self._db_connected = False
        
        print(f" DistributedRLTrainer {worker_id} initialized")
        print(f"  Database: {db_host}:{db_port}/{db_name}")
    
    async def _ensure_db_connection(self):
        """Ensure database is connected."""
        if not self._db_connected:
            await self.store.connect()
            self._db_connected = True
    
    async def train_episodes(
        self,
        episodes: int,
        query: str,
        start_paper_id: str,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.3,
        epsilon_decay: float = 0.995,
        **kwargs
    ) -> Dict:
        """
        Train RL agent for specified episodes with remote database.
        
        Args:
            episodes: Number of training episodes
            query: Search query for RL environment
            start_paper_id: Starting paper ID
            learning_rate: DDQN learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
        
        Returns:
            Dict with training results
        """
        start_time = time.time()
        
        print(f"\n[{self.worker_id}] Starting Distributed RL Training")
        print(f"  Episodes: {episodes}")
        print(f"  Query: {query[:50]}...")
        print(f"  Start paper: {start_paper_id}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        await self._ensure_db_connection()
        
        try:
            from RL.ddqn import DDQLAgent
            from RL.env import AdvancedGraphTraversalEnv
            
            env = AdvancedGraphTraversalEnv(
                store=self.store,
                use_communities=True,
                use_feedback=False 
            )
            
            # Create agent
            agent = DDQLAgent(
                state_dim=773,
                text_dim=384,
            )
            agent.epsilon = epsilon_start
            
            print(f"[{self.worker_id}] Agent initialized")
        
        except Exception as e:
            print(f"[{self.worker_id}] Failed to initialize agent: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'worker_id': self.worker_id
            }

        episode_rewards = []
        episode_similarities = []
        losses = []
        
        epsilon = epsilon_start
        
        for episode in range(episodes):
            try:
                state = await env.reset(
                    query=query,
                    intent=1,
                    start_node_id=start_paper_id
                )
                
                episode_reward = 0.0
                step = 0
                done = False
                
                while not done and step < env.max_steps:
                    step += 1
                    #Manager step 
                    manager_actions = await env.get_manager_actions()
                    if not manager_actions:
                        break
                    
                    manager_action = 1 if 1 in manager_actions else manager_actions[0]
                    is_terminal, manager_reward = await env.manager_step(manager_action)
                    episode_reward += manager_reward
                    
                    if is_terminal:
                        break
                    
                    # Worker step
                    worker_actions = await env.get_worker_actions()
                    if not worker_actions:
                        break
                    
                    best_action = agent.act(state, worker_actions)
                    if not best_action:
                        break
                    
                    chosen_node, _ = best_action
                    next_state, worker_reward, done = await env.worker_step(chosen_node)
                    episode_reward += worker_reward
                    
                    # Store experience
                    if not done:
                        next_manager_actions = await env.get_manager_actions()
                        if next_manager_actions:
                            temp_manager = 1 if 1 in next_manager_actions else next_manager_actions[0]
                            await env.manager_step(temp_manager)
                            next_worker_actions = await env.get_worker_actions()
                            env.pending_manager_action = None
                            env.available_worker_nodes = []
                        else:
                            next_worker_actions = []
                    else:
                        next_worker_actions = []
                    
                    agent.remember(
                        state=state,
                        action_tuple=best_action,
                        reward=episode_reward,
                        next_state=next_state,
                        done=done,
                        next_actions=next_worker_actions
                    )
                    
                    state = next_state
                
                # Record metrics
                summary = env.get_episode_summary()
                episode_rewards.append(episode_reward)
                episode_similarities.append(summary.get('max_similarity_achieved', 0.0))
                
                # Train agent
                if len(agent.memory) >= agent.batch_size:
                    if agent.use_prioritized:
                        loss = agent.replay_prioritized()
                    else:
                        loss = agent.replay()
                    losses.append(loss)
                else:
                    loss = 0.0
                
                # Decay epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                agent.epsilon = epsilon
                
                # Update target network
                if episode % 5 == 0 and episode > 0:
                    agent.update_target()
                
                # Progress logging
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_sim = np.mean(episode_similarities[-10:])
                    print(f"[{self.worker_id}] Episode {episode:4d}/{episodes} | "
                          f"Reward: {episode_reward:+7.2f} | "
                          f"Avg(10): {avg_reward:+7.2f} | "
                          f"Sim: {avg_sim:.3f} | "
                          f"Loss: {loss:.4f} | "
                          f"ε: {epsilon:.3f}")
            
            except Exception as e:
                print(f"[{self.worker_id}] Episode {episode} failed: {e}")
                continue
        
        duration = time.time() - start_time
        
        checkpoint_path = f"checkpoints/{self.worker_id}_final.pt"
        checkpoint = {
            'worker_id': self.worker_id,
            'episodes': episodes,
            'policy_net_state': agent.policy_net.state_dict(),
            'target_net_state': agent.target_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
            'epsilon': epsilon,
            'rewards': episode_rewards,
            'similarities': episode_similarities
        }
        torch.save(checkpoint, checkpoint_path)
        
        result = {
            'job_type': 'distributed_rl_training',
            'worker_id': self.worker_id,
            'episodes': episodes,
            'duration_sec': duration,
            'avg_reward': float(np.mean(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'final_reward': float(episode_rewards[-1]) if episode_rewards else 0.0,
            'avg_similarity': float(np.mean(episode_similarities)),
            'max_similarity': float(np.max(episode_similarities)),
            'final_similarity': float(episode_similarities[-1]) if episode_similarities else 0.0,
            'avg_loss': float(np.mean(losses)) if losses else 0.0,
            'final_epsilon': float(epsilon),
            'checkpoint_path': checkpoint_path,
            'status': 'completed'
        }
        
        print(f"\n[{self.worker_id}] Training Complete!")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")
        print(f"  Max Reward: {result['max_reward']:.2f}")
        print(f"  Avg Similarity: {result['avg_similarity']:.3f}")
        print(f"  Max Similarity: {result['max_similarity']:.3f}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        return result
    
    def __del__(self):
        """Cleanup database connection."""
        if self._db_connected:
            asyncio.run(self.store.disconnect())
