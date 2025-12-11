import asyncio
import numpy as np
import pickle
import os
import signal
import sys
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv
from graph.database.store import EnhancedStore


async def build_query_paper_pools(store, queries):
    """Build relevant paper pools for each query"""
    query_paper_pools = {}
    
    print("Building relevant paper pools...")
    for query in queries:
        pool = []
        keywords = query.split()
        
        for kw in keywords:
            if len(kw) < 4:
                continue
            papers = await store.get_paper_by_title(kw)
            pool.extend(papers[:10])
        
        # Deduplicate
        seen = set()
        unique_pool = []
        for p in pool:
            pid = p.get('paper_id')
            if pid and pid not in seen:
                seen.add(pid)
                unique_pool.append(p)
        
        query_paper_pools[query] = unique_pool[:30]
        print(f"  {query}: {len(unique_pool)} papers")
    
    return query_paper_pools


async def precompute_embeddings(papers, encoder):
    """Precompute embeddings for all papers"""
    embeddings = {}
    
    print(f"\nPrecomputing {len(papers)} embeddings...")
    for i, paper in enumerate(papers):
        pid = paper.get('paper_id')
        if pid and pid not in embeddings:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '') or ''
            text = f"{title} {abstract[:200]}"
            
            if text.strip():
                embeddings[pid] = encoder.encode(text)
            else:
                embeddings[pid] = encoder.encode(f"Paper {pid}")
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(papers)}")
    
    return embeddings


async def train():
    store = EnhancedStore()
    
    queries = [
        "attention mechanism transformers",
        "graph neural networks",
        "reinforcement learning optimization",
        "computer vision deep learning",
        "natural language processing"
    ]
    
    # Build query-specific paper pools
    query_paper_pools = await build_query_paper_pools(store, queries)
    
    # Precompute embeddings
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_papers = []
    for papers in query_paper_pools.values():
        all_papers.extend(papers)
    
    # Deduplicate all papers
    seen_pids = set()
    unique_papers = []
    for p in all_papers:
        pid = p.get('paper_id')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            unique_papers.append(p)
    
    embeddings = await precompute_embeddings(unique_papers, encoder)
    
    print(f"\nTotal unique papers: {len(unique_papers)}")
    print(f"Total embeddings: {len(embeddings)}")
    
    # Initialize environment and agent
    env = AdvancedGraphTraversalEnv(store, precomputedembeddings=embeddings)
    agent = DDQLAgent(state_dim=773, text_dim=384, precomputed_embeddings=embeddings)
    
    print(f"\nStarting training...")
    print(f"Warmup: {agent.warmup_steps}, Batch: {agent.batch_size}")
    print(f"=" * 70)
    
    episode_rewards = []
    episode_similarities = []
    
    for episode in range(500):
        query = np.random.choice(queries)
        paper_pool = query_paper_pools[query]
        
        if not paper_pool:
            print(f"[ERROR] No papers for query: {query}")
            continue
        
        start_paper = np.random.choice(paper_pool)
        
        try:
            state = await env.reset(query, intent=1, start_node_id=start_paper['paper_id'])
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            continue
        
        episode_reward = 0
        steps = 0
        dead_end_count = 0
        max_retries = 3
        
        # FIXED: Limit steps per episode to prevent hanging
        max_steps_per_episode = min(env.max_steps, 10)  # Cap at 10 steps
        
        for step in range(max_steps_per_episode):
            print(f"  [STEP] Episode {episode}, Step {step}")  # Progress tracking
            
            # Manager step
            manager_actions = await env.get_manager_actions()
            if not manager_actions:
                dead_end_count += 1
                if dead_end_count > max_retries:
                    print(f"  [BREAK] Dead end at step {step}")
                    break
                continue
            
            # Prefer CITEDBY to reduce dead ends
            if 1 in manager_actions and len(env.visited) > 2:
                manager_action = 1
            elif 1 in manager_actions:
                manager_action = 1
            else:
                manager_action = np.random.choice(manager_actions)
            
            is_terminal, manager_reward = await env.manager_step(manager_action)
            episode_reward += manager_reward
            
            if is_terminal:
                print(f"  [BREAK] Terminal at step {step}")
                break
            
            # Worker step
            worker_actions = await env.get_worker_actions()
            if not worker_actions:
                dead_end_count += 1
                if dead_end_count > max_retries:
                    print(f"  [SKIP] No worker actions at episode {episode}, step {step}")
                    break
                continue
            
            # FIXED: Limit worker actions to prevent slow Q-value computation
            if len(worker_actions) > 20:
                print(f"  [LIMIT] Sampling 20 from {len(worker_actions)} worker actions")
                worker_actions = worker_actions[:20]  # Just take first 20
            
            dead_end_count = 0
            
            print(f"  [ACT] Computing action for {len(worker_actions)} options...")
            best_action = agent.act(state, worker_actions)
            if not best_action:
                print(f"  [BREAK] No action selected at step {step}")
                break
            
            chosen_node, _ = best_action
            print(f"  [WORKER] Executing worker step...")
            next_state, worker_reward, done = await env.worker_step(chosen_node)
            episode_reward += worker_reward
            steps += 1
            
            if not done:
                next_actions = await env.get_worker_actions()
                # Limit next actions too
                if len(next_actions) > 20:
                    next_actions = next_actions[:20]
            else:
                next_actions = []
            
            print(f"  [REMEMBER] Storing transition...")
            agent.remember(
                state=state,
                action_tuple=best_action,
                reward=worker_reward,
                next_state=next_state,
                done=done,
                next_actions=next_actions
            )
            
            state = next_state
            
            if done:
                print(f"  [DONE] Episode complete at step {step}")
                break
        
        # Train every episode
        print(f"  [TRAIN] Training agent...")
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
            print(f"  [TRAIN] Loss: {loss:.4f}")
        else:
            loss = 0.0
            print(f"  [TRAIN] Skipped (memory: {len(agent.memory)}/{agent.batch_size})")
        
        # Update target network periodically
        if episode % 10 == 0 and episode > 0:
            print(f"  [TARGET] Updating target network...")
            agent.update_target()
        
        episode_rewards.append(episode_reward)
        episode_similarities.append(env.best_similarity_so_far)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_sim = np.mean(episode_similarities[-20:]) if episode_similarities else 0
            best_sim = max(episode_similarities) if episode_similarities else 0
            best_ep = episode_similarities.index(best_sim) if episode_similarities else 0
            
            print(f"\n{'='*70}")
            print(f"Episode {episode}/500")
            print(f"{'='*70}")
            print(f"  Reward: {episode_reward:+.2f} | Avg(20): {avg_reward:+.2f}")
            print(f"  Steps: {steps:2d} | RL Loss: {loss:.4f}")
            print(f"  Similarity: {env.best_similarity_so_far:.3f} | Avg(20): {avg_sim:.3f}")
            print(f"  Best Sim: {best_sim:.3f} (ep {best_ep})")
            print(f"  Epsilon: {agent.epsilon:.3f} | Memory: {len(agent.memory)}")
            print(f"  Total Steps: {agent.training_step}")
            print(f"{'='*70}\n")
        
        # Save checkpoints
        if episode % 50 == 0 and episode > 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(f'checkpoints/agent_ep{episode}.pt')
            print(f"[CHECKPOINT] Saved at episode {episode}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    agent.save('final_agent.pt')


if __name__ == '__main__':
    asyncio.run(train())
