import asyncio
import numpy as np
import pickle
import os
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv
from graph.database.store import EnhancedStore


async def sample_fresh_subgraph(store, query, encoder, embeddings, k=100):
    """Sample a fresh connected subgraph for each episode."""
    query_emb = encoder.encode_with_cache(query, cache_key=f"query_{query}")
    
    cypher = """
    MATCH (p:Paper)
    WHERE p.citationCount > 20
      AND p.year > 2015
      AND p.abstract IS NOT NULL
      AND size(p.title) > 10
    RETURN p.paperId as paper_id, p.title as title, p.abstract as abstract
    ORDER BY rand()
    LIMIT 500
    """
    
    candidates = await store._run_query_method(cypher)
    
    scores = []
    for paper in candidates:
        pid = paper['paper_id']
        
        if pid in embeddings:
            paper_emb = embeddings[pid]
        else:
            title = paper.get('title', '')
            abstract = (paper.get('abstract', '') or '')[:200]
            text = f"{title} {abstract}"
            paper_emb = encoder.encode_with_cache(text, cache_key=pid)
            embeddings[pid] = paper_emb
        
        sim = np.dot(query_emb, paper_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
        )
        scores.append((sim, paper))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[0][1] if scores else None


async def load_cached_data():
    """Load embeddings only (no query pools)."""
    print("Loading embeddings cache...")
    
    with open('embeddings_cache.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings):,} embeddings")
    
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
    
    # Load embeddings only
    embeddings , _  = await load_cached_data()
    
    # Initialize encoder for on-the-fly encoding
    from utils.batchencoder import BatchEncoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='embeddings_cache.pkl'
    )
    
    env = AdvancedGraphTraversalEnv(
        store, 
        precomputedembeddings=embeddings,
        use_communities=True,
        use_authors=True
    )
    
    agent = DDQLAgent(
        state_dim=773, 
        text_dim=384, 
        precomputed_embeddings=embeddings
    )
    
    agent.epsilon = 0.8
    agent.epsilon_decay = 0.998
    agent.epsilon_min = 0.05
    
    print(f"\n{'='*70}")
    print("TRAINING START")
    print(f"{'='*70}\n")
    
    episode_rewards = []
    episode_similarities = []
    
    for episode in range(500):
        query = np.random.choice(queries)
        
        start_paper = await sample_fresh_subgraph(store, query, encoder, embeddings)
        
        if not start_paper:
            print(f"[ERROR] Couldn't sample paper for query: {query}")
            continue
        
        start_id = start_paper['paper_id']
        
        if episode % 10 == 0:
            title = start_paper.get('title', 'Unknown')[:60]
            print(f"\n[EP {episode}] Query: '{query[:40]}'")
            print(f"[EP {episode}] Start: '{title}'")
        
        try:
            state = await env.reset(query, intent=1, start_node_id=start_id)
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            continue
        
        episode_reward = 0
        steps = 0
        max_steps_per_episode = 15  # Increased from 10
        
        for step in range(max_steps_per_episode):
            manager_actions = await env.get_manager_actions()
            
            if not manager_actions:
                if step == 0:
                    print(f"    [ERROR] No manager actions at start!")
                break
            
            # Prioritize citation-based exploration
            if 1 in manager_actions:  # CITEDBY
                manager_action = 1
            elif 0 in manager_actions:  # CITES
                manager_action = 0
            elif 2 in manager_actions:  # WROTE
                manager_action = 2
            elif 3 in manager_actions:  # AUTHORED
                manager_action = 3
            else:
                # Filter noisy relations
                if agent.epsilon > 0.5:
                    filtered = [a for a in manager_actions if a not in [6, 7]]
                    manager_action = np.random.choice(filtered if filtered else manager_actions)
                else:
                    manager_action = np.random.choice(manager_actions)
            
            is_terminal, manager_reward = await env.manager_step(manager_action)
            episode_reward += manager_reward
            
            if is_terminal:
                break
            
            worker_actions = await env.get_worker_actions()
            
            if not worker_actions:
                break
            
            # Limit action space
            if len(worker_actions) > 20:
                worker_actions = worker_actions[:20]
            
            best_action = agent.act(state, worker_actions)
            if not best_action:
                break
            
            chosen_node, _ = best_action
            next_state, worker_reward, done = await env.worker_step(chosen_node)
            episode_reward += worker_reward
            steps += 1
            
            next_actions = await env.get_worker_actions() if not done else []
            if len(next_actions) > 20:
                next_actions = next_actions[:20]
            
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
                break
        
        # Train
        loss = agent.replay() if len(agent.memory) >= agent.batch_size else 0.0
        
        if episode % 10 == 0 and episode > 0:
            agent.update_target()
        
        episode_rewards.append(episode_reward)
        episode_similarities.append(env.best_similarity_so_far)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            avg_sim = np.mean(episode_similarities[-20:]) if len(episode_similarities) >= 20 else np.mean(episode_similarities)
            best_sim = max(episode_similarities) if episode_similarities else 0
            
            print(f"\n{'='*70}")
            print(f"Episode {episode}/500")
            print(f"{'='*70}")
            print(f"  Reward: {episode_reward:+6.2f} | Avg(20): {avg_reward:+6.2f}")
            print(f"  Similarity: {env.best_similarity_so_far:.3f} | Avg: {avg_sim:.3f} | Best: {best_sim:.3f}")
            print(f"  Steps: {steps:2d} | Loss: {loss:.4f} | ε: {agent.epsilon:.3f} | Mem: {len(agent.memory)}")
            print(f"  Cache size: {len(embeddings):,}")
            print(f"{'='*70}\n")
        
        if episode % 50 == 0 and episode > 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(f'checkpoints/agent_ep{episode}.pt')
            print(f"[SAVE] Checkpoint at episode {episode}")
    
    print("\n✓ Training complete!")
    agent.save('final_agent.pt')


if __name__ == '__main__':
    asyncio.run(train())
