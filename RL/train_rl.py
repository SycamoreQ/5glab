import asyncio
import numpy as np
import pickle
import os
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
from RL.curriculum import CurriculumManager  

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed")

class WandBLogger:
    """Safe W&B wrapper with fallback."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
    def init(self, **kwargs):
        if not self.enabled:
            print("W&B disabled")
            return self
        
        try:
            self.run = wandb.init(**kwargs)
            print(f"W&B Run: {self.run.url}")
            return self.run
        except Exception as e:
            print(f"W&B init failed: {e}")
            self.enabled = False
            return self
    
    def log(self, metrics):
        if self.enabled and self.run:
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"[WARN] W&B log failed: {e}")
    
    def watch(self, model, **kwargs):
        if self.enabled and self.run:
            try:
                wandb.watch(model, **kwargs)
            except Exception as e:
                print(f"[WARN] W&B watch failed: {e}")
    
    def save(self, path):
        if self.enabled and self.run:
            try:
                wandb.save(path)
            except Exception as e:
                print(f"[WARN] W&B save failed: {e}")
    
    def finish(self):
        if self.enabled and self.run:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARN] W&B finish failed: {e}")
    
    @property
    def url(self):
        if self.run:
            return self.run.url
        return "N/A (W&B disabled)"


WANDB_PROJECT = "Enki"


CONFIG = {
    "total_episodes": 1000, 
    "max_steps_per_episode": 12,  
    "batch_size": 32,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    
    "epsilon_start": 1.0,
    "epsilon_min": 0.15,  
    "epsilon_decay": 0.9995, 
    "epsilon_warmup_episodes": 100,  
    "epsilon_curriculum_boost": 0.1, 
    
    "target_update_freq": 10,
    "use_communities": True,
    "state_dim": 783,
    "text_dim": 384,
    "use_prioritized_replay": True,
    
    # CURRICULUM SETTINGS
    "use_curriculum": True,
}

def normalize_paper_id(paper_id: str) -> str:
    """Normalize paper ID to consistent format."""
    if not paper_id:
        return ""
    paper_id = str(paper_id).strip().lstrip('0')
    return paper_id if paper_id else "0"

def get_available_cache_relations(env, pid: str):
    """Return list of RelationType enums that have edges in training cache."""
    pid = normalize_paper_id(pid)
    edges = env.training_edge_cache.get(pid, [])
    etypes = {et for et, _ in edges}
    actions = []
    if "cites" in etypes:
        actions.append(RelationType.CITES)
    if "citedby" in etypes:
        actions.append(RelationType.CITED_BY)
    return actions

def log_episode_metrics(logger, episode, episode_reward, steps, final_sim, loss, 
                        agent, env, query, curriculum_manager):
    """Log episode metrics to W&B with curriculum info."""
    stage = curriculum_manager.get_current_stage(episode)
    
    metrics = {
        "episode": episode,
        "episode_reward": episode_reward,
        "episode_steps": steps,
        "episode_similarity": final_sim,
        "episode_success": 1 if final_sim > 0.5 else 0,
        "epsilon": agent.epsilon,
        "loss": loss,
        "memory_size": len(agent.memory),
        "query_length": len(query.split()),
        
        # Curriculum tracking
        "curriculum_stage": curriculum_manager.current_stage,
        "curriculum_difficulty": stage['query_difficulty'],
        "max_steps_allowed": stage['max_steps'],
        "similarity_threshold": stage['start_similarity_threshold'],
    }
    
    # Community tracking
    if CONFIG['use_communities']:
        try:
            summary = env.get_episode_summary()
            metrics.update({
                "communities_visited": summary.get('unique_communities_visited', 0),
                "community_switches": summary.get('community_switches', 0),
                "max_steps_in_community": summary.get('max_steps_in_community', 0),
                "community_loops": summary.get('community_loops', 0),
            })
        except:
            pass
    
    logger.log(metrics)

def log_aggregate_metrics(logger, episode, episode_rewards, episode_similarities, 
                          episode_lengths, success_count, dead_end_count,
                          difficulty_stats):
    """Log aggregated metrics with per-difficulty breakdown."""
    window = min(50, len(episode_rewards))
    
    metrics = {
        "avg_reward_50": np.mean(episode_rewards[-window:]),
        "avg_similarity_50": float(np.nanmean(episode_similarities[-window:])),
        "avg_steps_50": np.mean(episode_lengths[-window:]),
        "best_similarity_overall": float(np.nanmax(episode_similarities)),
        "success_rate": 100 * success_count / (episode + 1),
        "dead_end_rate": 100 * dead_end_count / (episode + 1),
        "reward_std": np.std(episode_rewards[-window:]),
        "similarity_std": float(np.nanstd(episode_similarities[-window:])),
    }
    
    # Per-difficulty success rates
    for difficulty in ['easy', 'medium', 'hard', 'expert']:
        if difficulty in difficulty_stats and difficulty_stats[difficulty]['count'] > 0:
            success_rate = (100 * difficulty_stats[difficulty]['successes'] / 
                           difficulty_stats[difficulty]['count'])
            avg_sim = difficulty_stats[difficulty]['avg_similarity']
            metrics[f"success_rate_{difficulty}"] = success_rate
            metrics[f"avg_similarity_{difficulty}"] = avg_sim
    
    logger.log(metrics)

def log_trajectory(logger, env, episode):
    """Log detailed trajectory information - SAFE VERSION."""
    if not WANDB_AVAILABLE or not logger.enabled:
        return
    
    try:
        if not hasattr(env, 'trajectory_history') or len(env.trajectory_history) == 0:
            return
        
        trajectory_data = []
        for i, traj in enumerate(env.trajectory_history):
            node = traj.get('node', {})
            
            title = node.get('title') or node.get('name')
            if not title:
                paper_id = node.get('paper_id') or node.get('paperId')
                title = f"Paper {paper_id[-8:]}" if paper_id else "Unknown"
            
            title_str = str(title)[:50]
            
            trajectory_data.append([
                i,
                title_str,
                traj.get('similarity', 0.0),
                traj.get('reward', 0.0),
                str(traj.get('community', 'N/A'))
            ])
        
        table = wandb.Table(
            columns=["Step", "Paper", "Similarity", "Reward", "Community"],
            data=trajectory_data
        )
        logger.log({f"trajectory_ep{episode}": table})
        
    except Exception as e:
        print(f"[WARN] Failed to log trajectory: {e}")


async def load_training_cache():
    """Load cached training data."""
    print("Loading training cache...")
    cache_dir = 'training_cache'
    
    with open(os.path.join(cache_dir, 'training_papers_1M.pkl'), 'rb') as f:
        papers = pickle.load(f)
    print(f"✓ Loaded {len(papers):,} papers")
    
    with open(os.path.join(cache_dir, 'edge_cache_1M.pkl'), 'rb') as f:
        edge_cache = pickle.load(f)
    print(f"✓ Loaded edge cache")
    
    with open(os.path.join(cache_dir, 'paper_id_set_1M.pkl'), 'rb') as f:
        paper_id_set = pickle.load(f)
    print(f"✓ Loaded paper ID index")
    
    return papers, edge_cache, paper_id_set

async def build_embeddings():
    """Build or load embeddings."""
    papers, edge_cache, paper_id_set = await load_training_cache()
    print("Loading embeddings...")
    
    from utils.batchencoder import BatchEncoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'
    )
    
    encoder.precompute_paper_embeddings(papers, force=False)
    embeddings_raw = encoder.cache
    
    # Normalize IDs
    embeddings = {normalize_paper_id(str(k)): v for k, v in embeddings_raw.items()}
    
    # Normalize edge cache
    edge_cache_str = {}
    for src, edges in edge_cache.items():
        src_normalized = normalize_paper_id(str(src))
        normalized_edges = [
            (et, normalize_paper_id(str(tid)))
            for et, tid in edges
        ]
        edge_cache_str[src_normalized] = normalized_edges
    
    print(f"Loaded {len(embeddings):,} embeddings")
    return papers, edge_cache_str, paper_id_set, embeddings, encoder


async def train():
    """Main training loop with curriculum integration."""
    
    # Initialize W&B logger
    logger = WandBLogger(enabled=True)
    logger.init(
        project=WANDB_PROJECT,
        config=CONFIG,
        name=f"curriculum_rl_{time.strftime('%Y%m%d_%H%M%S')}",
        tags=["curriculum", "ddqn", "exploration-boost", "single-agent"],
        notes="Curriculum learning with improved exploration schedule"
    )
    
    # Load data
    papers, edge_cache, paper_id_set, embeddings, encoder = await build_embeddings()
    store = EnhancedStore(pool_size=10)
    
    # Initialize environment
    env = AdvancedGraphTraversalEnv(
        store,
        precomputed_embeddings=embeddings,
        use_communities=CONFIG['use_communities'],
        use_feedback=False,
        use_query_parser=True,
        parser_type='dspy',
        require_precomputed_embeddings=False
    )
    
    # Setup training cache
    embedded_ids = set(embeddings.keys())
    normalized_paper_id_set = {normalize_paper_id(str(pid)) for pid in paper_id_set}
    env.training_paper_ids = normalized_paper_id_set
    
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
    
    env.training_edge_cache = pruned_edge_cache
    env.precomputed_embeddings = embeddings
    
    print(f"\n✓ Graph statistics:")
    print(f"  Papers: {len(env.training_paper_ids):,}")
    print(f"  Edges: {sum(len(v) for v in env.training_edge_cache.values()):,}")
    
    # Initialize curriculum manager
    curriculum = CurriculumManager(papers, encoder)
    print("\n✓ Curriculum initialized:")
    print(f"  Stages: {len(curriculum.stages)}")
    print(f"  Total curriculum episodes: {sum(curriculum.stage_episodes)}")
    
    # Initialize agent
    agent = DDQLAgent(
        state_dim=CONFIG['state_dim'],
        text_dim=CONFIG['text_dim'],
        use_prioritized=CONFIG['use_prioritized_replay'],
        precomputed_embeddings=embeddings
    )
    
    # Watch model
    logger.watch(agent.policy_net, log="gradients", log_freq=100)
    
    print("\n" + "="*80)
    print("STARTING CURRICULUM TRAINING")
    print("="*80)
    print(f"W&B: {logger.url}")
    print(f"Total Episodes: {CONFIG['total_episodes']}")
    print(f"Epsilon: {CONFIG['epsilon_start']} → {CONFIG['epsilon_min']} (decay: {CONFIG['epsilon_decay']})")
    print("="*80 + "\n")
    
    # Training state
    episode_rewards = []
    episode_similarities = []
    episode_lengths = []
    dead_end_count = 0
    success_count = 0
    
    # Per-difficulty tracking
    difficulty_stats = defaultdict(lambda: {
        'count': 0, 
        'successes': 0, 
        'similarities': []
    })
    
    last_stage = -1
    
    # Training loop
    for episode in range(CONFIG['total_episodes']):
        try:
            stage = curriculum.get_current_stage(episode)
            current_stage_idx = curriculum.current_stage
            
            if current_stage_idx != last_stage:
                agent.epsilon = min(1.0, agent.epsilon + CONFIG['epsilon_curriculum_boost'])
                print(curriculum.get_stage_summary())
                print(f"[CURRICULUM] Stage transition! Epsilon boosted to {agent.epsilon:.3f}\n")
                last_stage = current_stage_idx

            if stage['query_difficulty'] == 'easy':
                env.max_revisits = 5
            elif stage['query_difficulty'] == 'medium':
                env.max_revisits = 4
            else:
                env.max_revisits = 3
            
            # Get query from curriculum
            query = curriculum.get_query_for_stage(stage, episode)
            difficulty = stage['query_difficulty']
            
            # Get starting paper from curriculum
            start_paper = curriculum.get_starting_paper(query, stage)
            start_paper_id = normalize_paper_id(
                str(start_paper.get('paperId') or start_paper.get('paper_id'))
            )
            
            # Validate starting paper
            if start_paper_id not in env.training_edge_cache:
                continue
            
            neighbor_ids = [tid for _, tid in env.training_edge_cache[start_paper_id]]
            if not any(nid in embedded_ids for nid in neighbor_ids):
                continue
            
            # Reset environment with stage-specific max steps
            state = await env.reset(query, intent=1, start_node_id=start_paper_id)
            max_steps = min(stage['max_steps'], CONFIG['max_steps_per_episode'])
            
            # Run episode
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Manager step
                pid = normalize_paper_id(
                    str(env.current_node.get("paperId") or env.current_node.get("paper_id"))
                )
                available = get_available_cache_relations(env, pid)
                
                if not available:
                    break
                
                relation_type = int(np.random.choice(available))
                is_terminal, manager_reward = await env.manager_step(relation_type)
                episode_reward += manager_reward
                
                if is_terminal:
                    break
                
                # Worker step
                worker_actions = await env.get_worker_actions()
                if not worker_actions:
                    break
                
                # Filter valid actions
                worker_actions = [
                    (n, r) for (n, r) in worker_actions
                    if normalize_paper_id(
                        str(n.get("paperid") or n.get("paperId") or n.get("paper_id"))
                    ) in embedded_ids
                ][:15]
                
                if not worker_actions:
                    break
                
                # Agent selects action
                best_action = agent.act(state, worker_actions, max_actions=15)
                if not best_action or not isinstance(best_action, tuple):
                    break
                
                chosen_node, chosen_relation = best_action
                
                # Execute action
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                steps += 1
                
                # Get next actions
                next_actions = await env.get_worker_actions() if not done else []
                next_actions = [
                    (n, r) for n, r in next_actions
                    if normalize_paper_id(str(n.get('paperId') or n.get('paper_id'))) in embedded_ids
                ][:15]
                
                # Store transition
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
            
            # Training
            loss = 0.0
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()

                if episode >= CONFIG['epsilon_warmup_episodes']:
                    agent.epsilon = max(
                        CONFIG['epsilon_min'], 
                        agent.epsilon * CONFIG['epsilon_decay']
                    )   

            
            # Target network update
            if episode % CONFIG['target_update_freq'] == 0 and episode > 0:
                agent.update_target()
            
            # Track stats
            if steps < 2:
                dead_end_count += 1
            
            final_sim = env.best_similarity_so_far
            if final_sim > 0.5:
                success_count += 1
                difficulty_stats[difficulty]['successes'] += 1
            
            difficulty_stats[difficulty]['count'] += 1
            difficulty_stats[difficulty]['similarities'].append(final_sim)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)
            
            # Update curriculum performance tracker
            curriculum.update_performance(episode_reward, final_sim)
            
            # Log to W&B
            log_episode_metrics(logger, episode, episode_reward, steps, final_sim, 
                                loss, agent, env, query, curriculum)
            
            # Aggregate logging every 10 episodes
            if (episode + 1) % 10 == 0:
                # Calculate per-difficulty stats
                for diff in difficulty_stats:
                    if difficulty_stats[diff]['count'] > 0:
                        difficulty_stats[diff]['avg_similarity'] = float(
                            np.nanmean(difficulty_stats[diff]['similarities'])
                        )
                
                log_aggregate_metrics(
                    logger, episode, episode_rewards, episode_similarities, 
                    episode_lengths, success_count, dead_end_count,
                    difficulty_stats
                )
                
                # Console output
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                avg_sim = float(np.nanmean(episode_similarities[-50:])) if len(episode_similarities) >= 50 else float(np.nanmean(episode_similarities))
                
                print(f"\n{'='*70}")
                print(f"Episode {episode+1}/{CONFIG['total_episodes']} | {stage['name']}")
                print(f"{'='*70}")
                print(f"  Reward: {episode_reward:+7.2f} | Avg: {avg_reward:+7.2f}")
                print(f"  Similarity: {final_sim:.3f} | Avg: {avg_sim:.3f}")
                print(f"  Steps: {steps:2d}/{max_steps} | ε: {agent.epsilon:.3f}")
                print(f"  Success rate: {100*success_count/(episode+1):.1f}%")
                print(f"  Per-difficulty success:")
                for diff in ['easy', 'medium', 'hard', 'expert']:
                    if diff in difficulty_stats and difficulty_stats[diff]['count'] > 0:
                        rate = 100 * difficulty_stats[diff]['successes'] / difficulty_stats[diff]['count']
                        avg = difficulty_stats[diff].get('avg_similarity', 0.0)
                        print(f"    {diff:8s}: {rate:5.1f}% (avg sim: {avg:.3f})")
                print(f"{'='*70}\n")
            
            # Log trajectory every 50 episodes
            if (episode + 1) % 50 == 0:
                log_trajectory(logger, env, episode + 1)
            
            # Checkpoint every 200 episodes
            if (episode + 1) % 200 == 0:
                os.makedirs('checkpoints', exist_ok=True)
                checkpoint_path = f'checkpoints/curriculum_agent_ep{episode+1}.pt'
                agent.save(checkpoint_path)
                logger.save(checkpoint_path)
                print(f"[CHECKPOINT] Saved at episode {episode+1}")
                
        except Exception as e:
            print(f"[ERROR] Episode {episode} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    agent.save('final_curriculum_agent.pt')
    logger.save('final_curriculum_agent.pt')
    
    # Final statistics 
    print("\n" + "="*80)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*80)
    print(f"W&B: {logger.url}")
    
    if len(episode_rewards) > 0:
        print(f"Overall success rate: {100*success_count/len(episode_rewards):.1f}%")
        print(f"Best similarity: {float(np.nanmax(episode_similarities)):.3f}")
        print(f"\nFinal per-difficulty performance:")
        for diff in ['easy', 'medium', 'hard', 'expert']:
            if diff in difficulty_stats and difficulty_stats[diff]['count'] > 0:
                rate = 100 * difficulty_stats[diff]['successes'] / difficulty_stats[diff]['count']
                avg = difficulty_stats[diff].get('avg_similarity', 0.0)
                print(f"  {diff:8s}: {rate:5.1f}% success | {avg:.3f} avg similarity")
    else:
        print("WARNING: No episodes completed successfully")
        print(f"Total episodes attempted: {CONFIG['total_episodes']}")
        print(f"All episodes failed - check paper ID field names and data format")
    
    print("="*80)
    
    await store.pool.close()
    logger.finish()

if __name__ == '__main__':
    asyncio.run(train())
