# train_simple.py
import asyncio
import pickle
import torch
import numpy as np
from graph.database.store import EnhancedStore
from RL.env import AdvancedGraphTraversalEnv
from RL.ddqn import DDQLAgent
import matplotlib.pyplot as plt
from collections import deque
from utils.userfeedback import UserFeedbackTracker 
import argparse

async def train_single_process():
    """Simple single-process training with DDQN agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--parser', type=str, default='dspy',
                       choices=['dspy', 'llm', 'rule'],
                       help='Query parser type')
    parser.add_argument('--episodes', type=int, default=1000)
    args = parser.parse_args()
    
    print("Loading cached papers...")
    with open('training_papers.pkl', 'rb') as f:
        cached_papers = pickle.load(f)
    print(f"✓ Loaded {len(cached_papers)} papers")

    print("\nStarting papers sample:")
    for i, p in enumerate(cached_papers[:3], 1):
        print(f"  {i}. {p['title'][:60]}")
        print(f"     Connectivity: {p['ref_count']} refs, {p['cite_count']} cites")
    
    print("\nInitializing environment...")
    store = EnhancedStore(pool_size=20)
    env = AdvancedGraphTraversalEnv(store, use_communities=True)
    
    state_dim = 773  
    text_dim = 384   
    
    agent = DDQLAgent(state_dim=state_dim, text_dim=text_dim)
    print(f"✓ Agent initialized (state_dim={state_dim}, text_dim={text_dim})")
    
    num_episodes = 1000
    target_update_freq = 10
    save_freq = 100
    
    episode_rewards = []
    episode_steps = []
    episode_similarities = []
    losses = []
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING: {num_episodes} episodes")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        paper = np.random.choice(cached_papers)
        
        try:
            state = await env.reset(
                query=paper['title'],
                intent=1,  
                start_node_id=paper['paper_id']
            )

            #print(f"\n[DEBUG] Starting paper: {paper['title'][:50]}")
            #print(f"[DEBUG] Paper ID: {paper['paper_id']}")
            #print(f"[DEBUG] Current node in env: {env.current_node.get('paper_id', 'NONE')}")
            
            episode_reward = 0
            step = 0
            done = False
            episode_experiences = []
            
            while not done and step < env.max_steps:
                step += 1
                
                # Get valid actions from environment
                manager_actions = await env.get_manager_actions()
                if not manager_actions:
                    print(f"  [Ep {episode} Step {step}] No manager actions, ending")
                    break
                
                # Manager step: choose relation type
                # For DDQN, we treat manager actions as part of the combined action space
                # But for simplicity, let's just pick CITED_BY (1) if available, else random
                if 1 in manager_actions:  
                    manager_action = 1
                else:
                    manager_action = np.random.choice(manager_actions)
                
                is_terminal, manager_reward = await env.manager_step(manager_action)
                episode_reward += manager_reward
                
                if is_terminal:
                    done = True
                    break
                
                worker_actions = await env.get_worker_actions()
                if not worker_actions:
                    print(f"  [Ep {episode} Step {step}] No worker actions, ending")
                    break
                
                # Agent selects best worker action (node, relation_type)
                best_action = agent.act(state, worker_actions)
                
                if best_action is None:
                    print(f"  [Ep {episode} Step {step}] Agent returned no action")
                    break
                
                chosen_node, _ = best_action
                
                # Execute worker action
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                
                # Get next valid actions (for DDQN target calculation)
                if not done:
                    next_manager_actions = await env.get_manager_actions()
                    if next_manager_actions:
                        # Peek at next worker actions
                        temp_manager = 1 if 1 in next_manager_actions else next_manager_actions[0]
                        await env.manager_step(temp_manager)
                        next_worker_actions = await env.get_worker_actions()
                        # Reset env state
                        env.pending_manager_action = None
                        env.available_worker_nodes = []
                    else:
                        next_worker_actions = []
                else:
                    next_worker_actions = []
                
                # Store experience
                agent.remember(
                    state=state,
                    action_tuple=best_action,
                    reward=episode_reward,
                    next_state=next_state,
                    done=done,
                    next_actions=next_worker_actions
                )
                
                state = next_state
            

            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            
            summary = env.get_episode_summary()
            episode_similarities.append(summary.get('max_similarity_achieved', 0.0))
            
            # Train agent every episode if enough experiences
            if len(agent.memory) >= agent.batch_size:
                batch = random.sample(list(agent.memory), agent.batch_size)
                loss = agent.replay_batch(batch)
                losses.append(loss)
                
                agent.epsilon = max(0.01, agent.epsilon * agent.epsilon_decay)
            else:
                loss = 0.0
            
            if episode % target_update_freq == 0 and episode > 0:
                agent.update_target()
                print(f"  [Ep {episode}] Target network updated")
            
            if episode % 10 == 0:
                env.feedback_tracker.save_feedback()
                total_clicks = sum(env.feedback_tracker.clicks.values())
                total_saves = sum(env.feedback_tracker.saves.values())
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
                avg_sim = np.mean(episode_similarities[-100:]) if len(episode_similarities) >= 100 else np.mean(episode_similarities)
                print(f"  Feedback: {total_clicks} clicks, {total_saves} saves")
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:+7.2f} | "
                      f"Avg(100): {avg_reward:+7.2f} | "
                      f"Steps: {step} | "
                      f"Loss: {loss:.4f} | "
                      f"ε: {agent.epsilon:.3f}")
                print(f"  Stats: Communities: {summary['unique_communities_visited']}, "
                      f"Sim: {summary['max_similarity_achieved']:.3f}, "
                      f"Loops: {summary['community_loops']}")
                
        
        except Exception as e:
            print(f"Episode {episode} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if episode % save_freq == 0 and episode > 0:
            checkpoint = {
                'episode': episode,
                'policy_net_state': agent.policy_net.state_dict(),
                'target_net_state': agent.target_net.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'rewards': episode_rewards,
                'steps': episode_steps,
                'similarities': episode_similarities
            }
            torch.save(checkpoint, f"checkpoint_ep{episode}.pt")
            print(f"Checkpoint saved: checkpoint_ep{episode}.pt")
    
    await store.pool.close()
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*70}\n")
    env.query_parser.print_stats()

    final_checkpoint = {
        'episode': num_episodes,
        'policy_net_state': agent.policy_net.state_dict(),
        'target_net_state': agent.target_net.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'rewards': episode_rewards,
        'steps': episode_steps,
        'similarities': episode_similarities
    }
    torch.save(final_checkpoint, "final_model.pt")
    print("Final model saved: final_model.pt")
    
    print(f"\n{'='*70}")
    print("TRAINING STATISTICS")
    print(f"{'='*70}")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Positive reward episodes: {sum(1 for r in episode_rewards if r > 0)} / {num_episodes} "
          f"({100 * sum(1 for r in episode_rewards if r > 0) / num_episodes:.1f}%)")
    print(f"Average steps per episode: {np.mean(episode_steps):.1f}")
    print(f"Average max similarity: {np.mean(episode_similarities):.3f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import random
    asyncio.run(train_single_process())
