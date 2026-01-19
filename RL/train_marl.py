import asyncio
import numpy as np
import pickle
import os
import time
import wandb
from typing import List, Dict

from RL.marl import MultiAgentTrainer, AgentExperience
from graph.database.store import EnhancedStore

from train_rl import (
    load_training_cache,
    build_embeddings,
    prepare_query_pools,
    normalize_paper_id,
    COMPLEX_QUERIES
)

WANDB_PROJECT = "marl_research_paper_navigation"
WANDB_ENTITY = "your-wandb-username"

CONFIG = {
    "num_agents": 4,
    "sync_frequency": 10,  
    "parallel_episodes_per_step": 4,  
    
    # Training
    "total_global_episodes": 500,
    "max_steps_per_episode": 12,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    
    # Environment
    "use_communities": True,
    "state_dim": 783,
    "text_dim": 384,
}

def log_parallel_episode(global_ep, experiences: List[AgentExperience]):
    """Log metrics from parallel episode runs."""
    metrics = {
        "global_episode": global_ep,
        
        # Per-agent metrics
        **{f"agent_{exp.agent_id}_reward": exp.reward for exp in experiences},
        **{f"agent_{exp.agent_id}_similarity": exp.similarity for exp in experiences},
        **{f"agent_{exp.agent_id}_steps": exp.steps for exp in experiences},
        
        # Aggregate metrics
        "mean_reward": np.mean([exp.reward for exp in experiences]),
        "mean_similarity": np.mean([exp.similarity for exp in experiences]),
        "mean_steps": np.mean([exp.steps for exp in experiences]),
        "max_reward": max([exp.reward for exp in experiences]),
        "max_similarity": max([exp.similarity for exp in experiences]),
    }
    
    wandb.log(metrics)

def log_trainer_stats(trainer: MultiAgentTrainer, global_ep: int):
    """Log shared buffer and agent coordination stats."""
    stats = trainer.get_statistics()
    
    metrics = {
        "global_episode": global_ep,
        "shared_buffer_size": stats['total_experiences'],
        "high_quality_experiences": stats['high_quality_experiences'],
        "best_reward_in_buffer": stats['best_reward'],
        "avg_reward_top5": stats['avg_reward_top5'],
        **{f"agent_{i}_episodes": count 
           for i, count in enumerate(stats['agent_episodes'])},
    }
    
    wandb.log(metrics)


async def train_marl():
    """Multi-agent training with W&B."""
    
    # Initialize W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=CONFIG,
        name=f"marl_{CONFIG['num_agents']}agents_{time.strftime('%Y%m%d_%H%M%S')}",
        tags=["hierarchical", "marl", "parallel", "shared-experience"],
        notes=f"{CONFIG['num_agents']} agents with shared experience buffer"
    )
    
    # Load data
    papers, edge_cache, paper_id_set, embeddings, encoder = await build_embeddings()
    query_pools = await prepare_query_pools(encoder, papers, paper_id_set)
    
    # Initialize multi-agent trainer
    store = EnhancedStore(pool_size=10)
    trainer = MultiAgentTrainer(
        store=store,
        num_agents=CONFIG['num_agents'],
        use_communities=CONFIG['use_communities']
    )
    
    # Setup training cache for all agents
    normalized_paper_id_set = {normalize_paper_id(str(pid)) for pid in paper_id_set}
    embedded_ids = set(embeddings.keys())
    
    # Prune edge cache
    pruned_edge_cache = {}
    for src, edges in edge_cache.items():
        src = normalize_paper_id(str(src))
        if src not in embedded_ids:
            continue
        
        kept = [
            (et.lower().replace("_", ""), normalize_paper_id(str(tid)))
            for et, tid in edges
            if et.lower().replace("_", "") in ("cites", "citedby") and
               normalize_paper_id(str(tid)) in embedded_ids
        ]
        
        if kept:
            pruned_edge_cache[src] = kept
    
    # Apply to all agent environments
    for agent_worker in trainer.agents:
        agent_worker.env.training_paper_ids = normalized_paper_id_set
        agent_worker.env.training_edge_cache = pruned_edge_cache
        agent_worker.env.precomputed_embeddings = embeddings
        agent_worker.agent.precomputed_embeddings = embeddings
    
    print(f"\n✓ Initialized {CONFIG['num_agents']} agents")
    print(f"  Papers: {len(normalized_paper_id_set):,}")
    print(f"  Edges: {sum(len(v) for v in pruned_edge_cache.values()):,}")
    
    print("STARTING MULTI-AGENT TRAINING")
    print(f"W&B Run: {run.url}")
    print(f"Agents: {CONFIG['num_agents']}")
    print(f"Global episodes: {CONFIG['total_global_episodes']}")
    print(f"Parallel episodes per step: {CONFIG['parallel_episodes_per_step']}")
    print("="*80 + "\n")
    
    # Training loop
    global_episode = 0
    
    while global_episode < CONFIG['total_global_episodes']:
        # Sample queries and starting papers for parallel execution
        queries = [np.random.choice(COMPLEX_QUERIES) for _ in range(CONFIG['parallel_episodes_per_step'])]
        start_papers = []
        
        for query in queries:
            paper_pool = query_pools[query]
            if paper_pool:
                start_idx = np.random.choice(min(10, len(paper_pool)))
                start_paper = paper_pool[start_idx]
                start_papers.append(start_paper)
            else:
                # Fallback
                start_papers.append(papers[0])
        
        # Run parallel episodes
        try:
            experiences = await trainer.train_parallel_episode(queries, start_papers)
            
            # Log parallel episode results
            log_parallel_episode(global_episode, experiences)
            
            # Train all agents
            avg_loss = trainer.train_all_agents(batch_size=CONFIG['batch_size'])
            wandb.log({"avg_loss": avg_loss, "global_episode": global_episode})
            
            # Synchronize agents periodically
            if global_episode % CONFIG['sync_frequency'] == 0 and global_episode > 0:
                trainer.sync()
                print(f"[SYNC] Agents synchronized at episode {global_episode}")
                wandb.log({"agent_sync": 1, "global_episode": global_episode})
            
            # Update target networks
            if global_episode % 20 == 0 and global_episode > 0:
                trainer.update_target_network()
            
            # Log trainer stats
            if global_episode % 10 == 0:
                log_trainer_stats(trainer, global_episode)
            
            # Console output
            if global_episode % 10 == 0:
                stats = trainer.get_statistics()
                print(f"\nGlobal Episode: {global_episode}/{CONFIG['total_global_episodes']}")
                print(f"  Mean reward: {np.mean([exp.reward for exp in experiences]):.2f}")
                print(f"  Mean similarity: {np.mean([exp.similarity for exp in experiences]):.3f}")
                print(f"  Shared buffer: {stats['total_experiences']:,} experiences")
                print(f"  Best reward: {stats['best_reward']:.2f}")
            
            # Checkpoint
            if global_episode % 100 == 0 and global_episode > 0:
                os.makedirs('checkpoints', exist_ok=True)
                
                # Save best agent
                best_agent_idx = max(
                    range(CONFIG['num_agents']),
                    key=lambda i: sum(
                        exp.reward for exp in trainer.shared_buffer.buffer
                        if exp.agent_id == i
                    )
                )
                
                checkpoint_path = f'checkpoints/marl_best_agent_ep{global_episode}.pt'
                trainer.agents[best_agent_idx].agent.save(checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"[CHECKPOINT] Saved best agent (#{best_agent_idx}) at episode {global_episode}")
            
            global_episode += CONFIG['parallel_episodes_per_step']
            
        except Exception as e:
            print(f"[ERROR] Parallel episode failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    best_agent_idx = max(
        range(CONFIG['num_agents']),
        key=lambda i: sum(
            exp.reward for exp in trainer.shared_buffer.buffer
            if exp.agent_id == i
        )
    )
    
    final_path = 'final_marl_agent.pt'
    trainer.agents[best_agent_idx].agent.save(final_path)
    wandb.save(final_path)
    
    print("\n" + "="*80)
    print("MULTI-AGENT TRAINING COMPLETE")
    print("="*80)
    print(f"W&B Run: {run.url}")
    print(f"Best agent: #{best_agent_idx}")
    print(f"Total experiences: {len(trainer.shared_buffer.buffer):,}")
    print("="*80)
    
    await store.pool.close()
    wandb.finish()

if __name__ == '__main__':
    asyncio.run(train_marl())
