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
import logging
from utils.batchencoder import BatchEncoder
import os 

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING) 

import sys
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


async def diagnose_communities(env, store):
    """Check if community detection is working."""
    print("Community Detection Diagnostic" )
    
    if not env.use_communities:
        print(" Community detection is DISABLED")
        print("="*70 + "\n")
        return
    
    print("Community detection is ENABLED")
    
    query = """
    MATCH (p:Paper)
    RETURN elementId(p) as paper_id, p.title as title
    LIMIT 5
    """
    
    try:
        sample_papers = await store._run_query_method(query)
        
        if not sample_papers:
            print("No papers found in database!")
            print("="*70 + "\n")
            return
        
        print(f"\nTesting {len(sample_papers)} sample papers:\n")
        
        found_count = 0
        for i, paper in enumerate(sample_papers, 1):
            paper_id = paper['paper_id']
            title = paper.get('title', 'No title')[:50]
            
            comm = env.community_detector.get_community(paper_id)
            
            print(f"{i}. Paper: {title}...")
            print(f"   ID: {paper_id}")
            print(f"   Community: {comm if comm else ' NOT FOUND'}")
            
            if comm:
                found_count += 1
                size = env.community_detector.get_community_size(comm)
                print(f"   Community size: {size} papers")
            else:
                print(f" No community mapping found for this ID!")
            print()
        
        print("Cache Statistics:")
        print(f"Papers in cache: {len(env.community_detector.paper_communities):,}")
        print(f"Authors in cache: {len(env.community_detector.author_communities):,}")
        print(f"Unique paper communities: {len(set(env.community_detector.paper_communities.values()))}")
        print(f"Unique author communities: {len(set(env.community_detector.author_communities.values()))}")
        
        print("\nSample cache ID formats:")
        sample_cache_ids = list(env.community_detector.paper_communities.keys())[:3]
        for cache_id in sample_cache_ids:
            comm = env.community_detector.paper_communities[cache_id]
            print(f"  {cache_id} → {comm}")
        
        print("\nSample Neo4j ID formats:")
        for paper in sample_papers[:3]:
            print(f"  {paper['paper_id']}")
        
        print("Analysis:")
        
        if found_count == 0:
            print("CRITICAL: No communities found for any papers!")
            print("\nPossible issues:")
            print("  1. ID format mismatch between cache and Neo4j")
            print("     - Cache uses one format, Neo4j returns another")
            print("  2. Community cache is outdated")
            print("     - Papers in DB don't match papers in cache")
            print("\nRecommended fix:")
            print("  Run: python -m RL.community_detection")
            print("  This will rebuild the cache with current paper IDs")
        elif found_count < len(sample_papers):
            print(f" WARNING: Only {found_count}/{len(sample_papers)} papers have communities")
            print("  Some papers might be missing from cache")
        else:
            print(f"SUCCESS: All {found_count}/{len(sample_papers)} test papers have communities!")
            print(" Community detection should work correctly")
        
    except Exception as e:
        print(f"Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()



async def train_single_process():
    """Simple single-process training with DDQN agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--parser', type=str, default='dspy',
                       choices=['dspy', 'llm', 'rule'],
                       help='Query parser type')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--skip-diagnostic', action='store_true',
                       help='Skip community diagnostic check')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
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
    encoder = BatchEncoder()
    precomputed_embeddings = encoder.precompute_paper_embeddings(cached_papers)
    env = AdvancedGraphTraversalEnv(store, use_communities=True)
    
    if not args.skip_diagnostic:
        await diagnose_communities(env, store)
        
        print("\nDo you want to continue training? (y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("Training cancelled. Fix issues and re-run.")
            await store.pool.close()
            return
        print()
    
    state_dim = 773  
    text_dim = 384   
    
    agent = DDQLAgent(state_dim=state_dim, text_dim=text_dim)
    agent.use_prioritized = True 
    print(f"Agent initialized (state_dim={state_dim}, text_dim={text_dim})")
    
    num_episodes = args.episodes
    target_update_freq = 5
    save_freq = 100
    
    episode_rewards = []
    episode_steps = []
    episode_similarities = []
    losses = []
    
    print(f"Starting Training for: {num_episodes} episodes")
    
    for episode in range(num_episodes):
        paper = np.random.choice(cached_papers)
        
        try:
            state = await env.reset(
                query=paper['title'],
                intent=1,  
                start_node_id=paper['paper_id']
            )
            
            episode_reward = 0
            step = 0
            done = False
            episode_experiences = []
            
            while not done and step < env.max_steps:
                step += 1
                
                manager_actions = await env.get_manager_actions()
                if not manager_actions:
                    if episode < 10: 
                        print(f"  [Ep {episode} Step {step}] No manager actions, ending")
                    break
                
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
                    if episode < 10: 
                        print(f"  [Ep {episode} Step {step}] No worker actions, ending")
                    break
                
                best_action = agent.act(state, worker_actions)
                
                if best_action is None:
                    if episode < 10:
                        print(f"  [Ep {episode} Step {step}] Agent returned no action")
                    break
                
                chosen_node, _ = best_action
                
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                
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
            
            if len(agent.memory) >= agent.batch_size:
                if agent.use_prioritized: 
                    loss = agent.replay_prioritized()
                else: 
                    loss = agent.replay()
                
                losses.append(loss)
            else:
                loss = 0.0
            
            if episode % target_update_freq == 0 and episode > 0:
                agent.update_target()
                print(f"  [Ep {episode}] Target network updated")
            
            # Print progress every 10 episodes
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
            logging.error(f"Episode {episode} failed: {e}")
            if episode < 5:  
                import traceback
                traceback.print_exc()
            continue

        # Save checkpoints
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
            print(f"  Checkpoint saved: checkpoint_ep{episode}.pt")
    
    await store.pool.close()
    
    print("Training Complete")
    
    env.query_parser.print_stats()

    # Save final model
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
    print("\n✓ Final model saved: final_model.pt")
    
    # Print statistics
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
