import ray 
import torch 
import asyncio
import logging
import time 
import os 
from collections import deque
import numpy as np 
from .env import AdvancedGraphTraversalEnv, RelationType
from .ddqn import DDQLAgent
from graph.database.store import EnhancedStore
import random 
from typing import List, Tuple, Dict, Any, Optional

@ray.remote(num_cpus=0.1)
class SharedStorage:
    def __init__(self):
        self.memory = deque(maxlen=500000)
        self.weights = None 
        self.target_weights = None
        self.episode_rewards = []
        self.episode_count = 0

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
    
    def log_reward(self, reward):
        self.episode_count += 1
        self.episode_rewards.append(reward)
        
        # Print more informative stats
        if self.episode_count % 10 == 0:
            recent_rewards = self.episode_rewards[-10:]
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {self.episode_count} | Reward: {reward:.4f} | Avg(last 10): {avg_recent:.4f} | Memory: {len(self.memory)}")
        else:
            print(f"Episode {self.episode_count} | Reward: {reward:.4f}")
        
    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)
    
    def get_stats(self):
        return {
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
        }


@ray.remote(num_cpus=1) 
class Explorer:
    def __init__(self, worker_id, storage_actor):
        self.worker_id = worker_id
        self.store = EnhancedStore() 
        self.env = AdvancedGraphTraversalEnv(self.store)
        self.agent = DDQLAgent(773, 384)
        self.storage = storage_actor

    async def _select_action_with_beam_search(self, state: np.ndarray, worker_actions: List[Tuple[Dict, int]]) -> Optional[Tuple[Dict, int]]:
        """
        Performs greedy action selection based on Q-values.
        Returns: (node_dict, rel_type) tuple or None
        """
        if not worker_actions:
            return None
            
        best_q_value = -float('inf')
        best_action_tuple = None

        for node_dict, rel_type in worker_actions:
            # Extract text from node - use multiple fallbacks
            node_text = (
                node_dict.get('title') or 
                node_dict.get('name') or 
                node_dict.get('doi') or 
                node_dict.get('original_id') or
                node_dict.get('paper_id') or
                ''
            )
            
            if not node_text:
                continue
                
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
        final_node_txt = final_node.get('title') or final_node.get('name') or final_node.get('doi') or 'final'
        hindsight_query_emb = self.agent.encoder.encode(final_node_txt)
        
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
            
            # Bonus for reaching final state
            if i == len(trajectory) - 1:
                new_total_reward += 5.0 
            
            h_exp = (new_state, action_tuple, new_total_reward, new_next_state, done, worker_actions)
            
            self.storage.add_experience.remote(h_exp)

    async def run_episode(self, episode_idx: int): 
        """
        Runs a single episode of hierarchical exploration.
        """
        try:
            # Get latest weights from learner
            weights = await self.storage.get_weights.remote()
            
            if weights: 
                self.agent.policy_net.load_state_dict(weights)
                self.agent.epsilon = max(0.05, 0.995**episode_idx)  # Slower decay

            # Reset environment - Get a well-connected paper from database
            initial_intent = RelationType.CITED_BY
            
            try:
                # Get a well-connected paper (has both citations and references)
                print(f"[Explorer {self.worker_id}] Fetching a well-connected paper...")
                paper = await self.store.get_well_connected_paper()
                
                if not paper:
                    print(f"[Explorer {self.worker_id}] No well-connected papers, using any paper...")
                    paper = await self.store.get_any_paper()
                
                if not paper:
                    raise ValueError("Database has no papers!")
                
                paper_id = paper.get('paper_id')
                if not paper_id:
                    raise ValueError("Paper has no paper_id field!")
                
                state = await self.env.reset(paper_id, initial_intent, start_node_id=paper_id)
                
                paper_title = paper.get('title', paper.get('original_id', 'Unknown'))[:60]
                ref_count = paper.get('ref_count', 0)
                cite_count = paper.get('cite_count', 0)
                
                print(f"[Explorer {self.worker_id}] Episode {episode_idx} started")
                print(f"  Paper: {paper_title}...")
                print(f"  Connectivity: {ref_count} refs, {cite_count} citations")
                
            except Exception as e:
                print(f"[Explorer {self.worker_id}] Failed to initialize: {e}")
                import traceback
                traceback.print_exc()
                raise ValueError("Cannot find any valid paper in database!")

            done = False 
            total_reward = 0.0
            local_memory = []
            step_count = 0
            
            print(f"[Ep {episode_idx}] Starting node: {self.env.current_node.get('title', 'Unknown')[:50]}")
            
            while not done and step_count < 10:  # Safety limit
                step_count += 1
                
                # MANAGER: Choose relation type
                valid_manager_actions = await self.env.get_manager_actions()
                
                if not valid_manager_actions:
                    print(f"[Ep {episode_idx}] No manager actions available. Ending episode.")
                    break
                
                # DEBUG: Force exploration for first few steps
                if step_count <= 2 and RelationType.STOP in valid_manager_actions:
                    if len(valid_manager_actions) > 1:
                        valid_manager_actions.remove(RelationType.STOP)
                
                # Epsilon-greedy for manager (simplified - just random for now)
                manager_action_reltype = random.choice(valid_manager_actions)

                is_terminal, manager_reward = await self.env.manager_step(manager_action_reltype)
                
                print(f"[Ep {episode_idx} Step {step_count}] Manager chose: {manager_action_reltype} | Reward: {manager_reward:.4f} | Terminal: {is_terminal}")
                
                if is_terminal: 
                    # Create terminal experience
                    action_emb = np.zeros(self.agent.text_dim)
                    action_tuple = (action_emb, manager_action_reltype)
                    experience = (state, action_tuple, manager_reward, state, True, [])
                    local_memory.append(experience) 
                    total_reward += manager_reward
                    done = True 
                    break

                # WORKER: Choose specific node
                worker_actions = await self.env.get_worker_actions()

                if not worker_actions:
                    print(f"[Ep {episode_idx} Step {step_count}] No worker actions available after manager chose {manager_action_reltype}. Continuing...")
                    # Add experience with negative reward for dead-end
                    action_emb = np.zeros(self.agent.text_dim)
                    action_tuple = (action_emb, manager_action_reltype)
                    experience = (state, action_tuple, manager_reward - 0.5, state, False, [])
                    local_memory.append(experience)
                    total_reward += (manager_reward - 0.5)
                    continue

                print(f"[Ep {episode_idx} Step {step_count}] {len(worker_actions)} worker actions available")

                # Worker action selection: epsilon-greedy
                if random.random() < self.agent.epsilon:
                    chosen_node_dict, chosen_rel_type = random.choice(worker_actions)
                    selection_method = "random"
                else:
                    best_action_tuple = await self._select_action_with_beam_search(state, worker_actions)
                    
                    if best_action_tuple:
                        chosen_node_dict, chosen_rel_type = best_action_tuple
                        selection_method = "greedy"
                    else:
                        chosen_node_dict, chosen_rel_type = random.choice(worker_actions)
                        selection_method = "fallback"

                chosen_title = (
                    chosen_node_dict.get('title') or 
                    chosen_node_dict.get('name') or 
                    chosen_node_dict.get('doi') or 
                    chosen_node_dict.get('original_id') or
                    chosen_node_dict.get('paper_id', '[Unknown]')
                )
                print(f"[Ep {episode_idx} Step {step_count}] Worker chose ({selection_method}): {chosen_title[:60]}")

                # Encode chosen action
                action_text = (
                    chosen_node_dict.get('title') or 
                    chosen_node_dict.get('name') or 
                    chosen_node_dict.get('doi') or 
                    chosen_node_dict.get('original_id') or
                    chosen_node_dict.get('paper_id') or
                    'empty'
                )
                action_emb = self.agent.encoder.encode(action_text)
                action_tuple = (action_emb, manager_action_reltype) 

                # Execute worker step
                next_state, worker_reward, done = await self.env.worker_step(chosen_node_dict)
                
                current_total_reward = manager_reward + worker_reward
                total_reward += current_total_reward

                print(f"[Ep {episode_idx} Step {step_count}] Worker reward: {worker_reward:.4f} | Total step reward: {current_total_reward:.4f} | Done: {done}")

                # Store experience
                experience = (state, action_tuple, current_total_reward, next_state, done, worker_actions)
                local_memory.append(experience)
                
                state = next_state

            # Add experiences to shared storage
            print(f"[Ep {episode_idx}] Episode finished. Total reward: {total_reward:.4f} | Steps: {step_count} | Experiences: {len(local_memory)}")
            
            for exp in local_memory: 
                self.storage.add_experience.remote(exp)
                
            # Generate HER experiences
            if len(local_memory) > 0: 
                await self._process_hindsight_experience(local_memory)
                    
            await self.storage.log_reward.remote(total_reward)

            return episode_idx, total_reward, local_memory
            
        except Exception as e:
            print(f"[Explorer {self.worker_id}] Episode {episode_idx} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return episode_idx, 0.0, []


@ray.remote(num_cpus=1)  # Changed from num_gpus=0.5 since GPU not available
class Learner:
    def __init__(self, storage_actor):
        self.agent = DDQLAgent(773, 384)
        self.storage = storage_actor 
        self.steps = 0 
        self.batch_size = 32
        self.update_frequency = 50
        self.target_update_frequency = 200

    async def update_model(self):
        """
        Continuous learning loop that samples from replay buffer and updates the model.
        """
        print("[Learner] Starting continuous learning loop...")
        
        # Set initial weights
        await self.storage.set_weights.remote(
            {k: v.cpu() for k, v in self.agent.policy_net.state_dict().items()}
        )

        while True:
            # Get batch from storage
            batch = await self.storage.get_batch.remote(self.batch_size)
            
            if not batch:
                # Wait for more experiences
                await asyncio.sleep(1)
                continue

            # Train on batch
            loss = self.agent.replay_batch(batch) 

            self.steps += 1

            # Periodically sync weights to storage
            if self.steps % self.update_frequency == 0:
                await self.storage.set_weights.remote(
                    {k: v.cpu() for k, v in self.agent.policy_net.state_dict().items()}
                )
                
                stats = await self.storage.get_stats.remote()
                print(f"[Learner] Step {self.steps} | Loss: {loss:.4f} | Memory: {stats['memory_size']} | Avg Reward: {stats['avg_reward']:.4f}")

            # Periodically update target network
            if self.steps % self.target_update_frequency == 0: 
                self.agent.update_target()
                print(f"[Learner] Step {self.steps} | Updated target network")
                
            # Save checkpoint periodically
            if self.steps % 500 == 0:
                os.makedirs("models", exist_ok=True)
                torch.save(
                    self.agent.policy_net.state_dict(),
                    f"models/ddqn_learner_step_{self.steps}.pth"
                )
                print(f"[Learner] Saved checkpoint at step {self.steps}")


def run_training(num_explorers: int = 4, total_eps: int = 2000): 
    """
    Main training loop using Ray for distributed RL.
    """
    if ray.is_initialized():
        ray.shutdown()

    context = ray.init(ignore_reinit_error=True)
    print(f'Ray initialized. Dashboard URL: {context.dashboard_url}')
    
    TOTAL_EPISODES = total_eps
    NUM_EXPLORERS = num_explorers

    # Create shared storage actor
    storage_actor = SharedStorage.remote()
    print("[Main] Storage actor created")

    # Create explorer actors
    explorer_actors = [Explorer.remote(i, storage_actor) for i in range(NUM_EXPLORERS)]
    print(f"[Main] {NUM_EXPLORERS} Explorer actors created")
    
    # Create and start learner actor
    learner_actor = Learner.remote(storage_actor)
    learner_future = learner_actor.update_model.remote()  # Start learning loop in background
    print("[Main] Learner actor created and started")

    # Launch episodes
    episode_futures = []

    for episode_idx in range(1, TOTAL_EPISODES + 1):
        explorer_id = (episode_idx - 1) % NUM_EXPLORERS
        explorer = explorer_actors[explorer_id]

        future = explorer.run_episode.remote(episode_idx)
        episode_futures.append(future)
        
        if episode_idx % 50 == 0:
            print(f"[Main] Launched {episode_idx} / {TOTAL_EPISODES} episodes")
            
        # Rate limiting to avoid overwhelming the system
        if episode_idx % 100 == 0:
            time.sleep(2)

    print("\n[Main] All exploration tasks launched. Waiting for completion...")
    
    # Wait for all episodes to complete
    results = ray.get(episode_futures)
    
    print("\n=== Training Completed ===")
    print(f"Total Episodes Run: {len(results)}")
    
    # Calculate statistics
    total_rewards = [r[1] for r in results if r and len(r) > 1]
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        max_reward = max(total_rewards)
        min_reward = min(total_rewards)
        
        print(f"Average Episode Reward: {avg_reward:.4f}")
        print(f"Max Episode Reward: {max_reward:.4f}")
        print(f"Min Episode Reward: {min_reward:.4f}")
        
        # Print reward distribution
        positive_rewards = sum(1 for r in total_rewards if r > 0)
        print(f"Episodes with positive reward: {positive_rewards} / {len(total_rewards)} ({100*positive_rewards/len(total_rewards):.1f}%)")
    
    # Get final stats from storage
    final_stats = ray.get(storage_actor.get_stats.remote())
    print(f"\nFinal Memory Size: {final_stats['memory_size']}")
    
    # Save final model
    print("\n[Main] Saving final model...")
    os.makedirs("models", exist_ok=True)
    
    ray.shutdown()
    print("[Main] Ray shutdown complete")


if __name__ == "__main__":
    # Run training with specified parameters
    run_training(num_explorers=4, total_eps=100)  # Start with fewer episodes for testing