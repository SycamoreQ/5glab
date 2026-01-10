import asyncio
import numpy as np
import pickle
import os
import time
import yaml 
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
from RL.curriculum import CurriculumManager
import ray
from ray import train
from ray.train import Checkpoint

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

DEFAULT_CONFIG = {
    "total_episodes": 1000,
    "max_steps_per_episode": 12,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.15,
    "epsilon_decay": 0.99985,
    "epsilon_warmup_episodes": 100,
    "epsilon_curriculum_boost": 0.1,
    "target_update_freq": 10,
    "use_communities": True,
    "state_dim": 783,
    "text_dim": 384,
    "use_prioritized_replay": True,
    "train_every_n_steps": 2,
    "end_of_episode_replays": 3,
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

class DistributedRLTrainer:

    def __init__(self, worker_id: str, db_host: str, db_port: int, 
                 db_user: str, db_password: str, db_name: str):
        self.worker_id = worker_id
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'user': db_user,
            'password': db_password,
            'database': db_name
        }
        self._initialized = False
        print(f"DistributedRLTrainer {worker_id} initialized")
        print(f"Database: {db_host}:{db_port}/{db_name}")

    async def _initialize_training_environment(self):
        """Initialize training environment (called once)."""
        if self._initialized:
            return

        print(f"[{self.worker_id}] Initializing training environment...")

        # Load data
        self.papers, self.edge_cache, self.paper_id_set, self.embeddings, self.encoder = await build_embeddings()

        self.store = EnhancedStore(pool_size=5)

        # Initialize environment
        self.env = AdvancedGraphTraversalEnv(
            self.store,
            precomputed_embeddings=self.embeddings,
            use_communities=DEFAULT_CONFIG['use_communities'],
            use_feedback=False,
            use_query_parser=True,
            parser_type='dspy',
            require_precomputed_embeddings=False
        )

        # Setup training cache
        embedded_ids = set(self.embeddings.keys())
        normalized_paper_id_set = {normalize_paper_id(str(pid)) for pid in self.paper_id_set}
        self.env.training_paper_ids = normalized_paper_id_set

        # Prune edge cache
        pruned_edge_cache = {}
        for src, edges in self.edge_cache.items():
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

        self.env.training_edge_cache = pruned_edge_cache
        self.env.precomputed_embeddings = self.embeddings
        self.embedded_ids = embedded_ids

        print(f"[{self.worker_id}] ✓ Environment initialized")
        print(f"  Papers: {len(self.env.training_paper_ids):,}")
        print(f"  Edges: {sum(len(v) for v in self.env.training_edge_cache.values()):,}")

        self._initialized = True

    async def train_episodes(self, episodes: int, query: str, start_paper_id: str, **config):
        CONFIG = {**DEFAULT_CONFIG, **config}
        CONFIG['total_episodes'] = episodes

        await self._initialize_training_environment()

        # Initialize WandB logger
        logger = WandBLogger(enabled=CONFIG.get('use_wandb', False))
        if logger.enabled:
            logger.init(
                project=WANDB_PROJECT,
                config=CONFIG,
                name=f"{self.worker_id}_{time.strftime('%Y%m%d_%H%M%S')}",
                tags=["distributed", "ddqn", self.worker_id],
                notes=f"Distributed training on {self.worker_id}"
            )

        # Initialize curriculum manager
        curriculum = CurriculumManager(self.papers, self.encoder)
        print(f"[{self.worker_id}] Curriculum initialized with {len(curriculum.stages)} stages")

        # Initialize agent
        agent = DDQLAgent(
            state_dim=CONFIG['state_dim'],
            text_dim=CONFIG['text_dim'],
            use_prioritized=CONFIG['use_prioritized_replay'],
            precomputed_embeddings=self.embeddings
        )

        if logger.enabled:
            logger.watch(agent.policy_net, log="gradients", log_freq=100)

        print(f"\n{'='*80}")
        print(f"[{self.worker_id}] STARTING TRAINING")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Episodes: {episodes}")
        print(f"Start Paper: {start_paper_id}")
        print(f"W&B: {logger.url}")
        print(f"{'='*80}\n")

        # Training state
        episode_rewards = []
        episode_similarities = []
        episode_lengths = []
        dead_end_count = 0
        success_count = 0
        total_training_steps = 0

        # Main training loop
        for episode in range(episodes):
            try:
                stage = curriculum.get_current_stage(episode)

                # Get query and starting paper
                if episode == 0:
                    # Use provided query and start paper for first episode
                    episode_query = query
                    episode_start_paper_id = normalize_paper_id(start_paper_id)
                else:
                    # Use curriculum for subsequent episodes
                    episode_query = curriculum.get_query_for_stage(stage, episode)
                    start_paper = curriculum.get_starting_paper(episode_query, stage)
                    episode_start_paper_id = normalize_paper_id(
                        str(start_paper.get('paperId') or start_paper.get('paper_id'))
                    )

                # Validate starting paper
                if episode_start_paper_id not in self.env.training_edge_cache:
                    continue

                neighbor_ids = [tid for _, tid in self.env.training_edge_cache[episode_start_paper_id]]
                if not any(nid in self.embedded_ids for nid in neighbor_ids):
                    continue

                # Reset environment
                state = await self.env.reset(episode_query, intent=1, start_node_id=episode_start_paper_id)
                max_steps = min(stage['max_steps'], CONFIG['max_steps_per_episode'])

                # Run episode
                episode_reward = 0
                steps = 0
                step_losses = []

                for step in range(max_steps):
                    # Manager step
                    pid = normalize_paper_id(
                        str(self.env.current_node.get("paperId") or self.env.current_node.get("paper_id"))
                    )
                    available = get_available_cache_relations(self.env, pid)
                    if not available:
                        break

                    relation_type = int(np.random.choice(available))
                    is_terminal, manager_reward = await self.env.manager_step(relation_type)
                    episode_reward += manager_reward

                    if is_terminal:
                        break

                    # Worker step
                    worker_actions = await self.env.get_worker_actions()
                    if not worker_actions:
                        break

                    # Filter valid actions
                    worker_actions = [
                        (n, r) for (n, r) in worker_actions
                        if normalize_paper_id(
                            str(n.get("paperid") or n.get("paperId") or n.get("paper_id"))
                        ) in self.embedded_ids
                    ][:15]

                    if not worker_actions:
                        break

                    # Agent selects action
                    best_action = agent.act(state, worker_actions, max_actions=15)
                    if not best_action or not isinstance(best_action, tuple):
                        break

                    chosen_node, chosen_relation = best_action

                    # Execute action
                    next_state, worker_reward, done = await self.env.worker_step(chosen_node)
                    exploration_bonus = 0.5 * (steps / max_steps) if steps >= 5 else 0.0
                    total_reward = worker_reward + exploration_bonus
                    episode_reward += total_reward
                    steps += 1

                    # Get next actions
                    next_actions = await self.env.get_worker_actions() if not done else []
                    next_actions = [
                        (n, r) for n, r in next_actions
                        if normalize_paper_id(str(n.get('paperId') or n.get('paper_id'))) in self.embedded_ids
                    ][:15]

                    # Store transition
                    agent.remember(
                        state=state,
                        action_tuple=best_action,
                        reward=total_reward,
                        next_state=next_state,
                        done=done,
                        next_actions=next_actions
                    )

                    # Train agent
                    if len(agent.memory) >= agent.batch_size and step % CONFIG['train_every_n_steps'] == 0:
                        loss = agent.replay()
                        step_losses.append(loss)
                        total_training_steps += 1

                        # Decay epsilon
                        if episode >= CONFIG['epsilon_warmup_episodes']:
                            agent.epsilon = max(
                                CONFIG['epsilon_min'],
                                agent.epsilon * CONFIG['epsilon_decay']
                            )

                    state = next_state

                    if done:
                        break

                # End of episode training
                if len(agent.memory) >= agent.batch_size:
                    for _ in range(CONFIG['end_of_episode_replays']):
                        loss = agent.replay()
                        step_losses.append(loss)
                        total_training_steps += 1

                        if episode >= CONFIG['epsilon_warmup_episodes']:
                            agent.epsilon = max(
                                CONFIG['epsilon_min'],
                                agent.epsilon * CONFIG['epsilon_decay']
                            )

                # Calculate metrics
                episode_loss = np.mean(step_losses) if step_losses else 0.0

                # Target network update
                if episode % CONFIG['target_update_freq'] == 0 and episode > 0:
                    agent.update_target()

                # Track stats
                if steps < 2:
                    dead_end_count += 1

                final_sim = self.env.best_similarity_so_far
                if final_sim > 0.5:
                    success_count += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)

                # Update curriculum
                curriculum.update_performance(episode_reward, final_sim)

                # Log metrics
                if logger.enabled:
                    logger.log({
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "episode_steps": steps,
                        "episode_similarity": final_sim,
                        "epsilon": agent.epsilon,
                        "loss": episode_loss,
                        "total_training_steps": total_training_steps,
                    })

                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-50:])
                    avg_sim = float(np.nanmean(episode_similarities[-50:]))
                    print(f"[{self.worker_id}] Ep {episode+1}/{episodes} | "
                          f"Reward: {episode_reward:+7.2f} (avg: {avg_reward:+7.2f}) | "
                          f"Sim: {final_sim:.3f} (avg: {avg_sim:.3f}) | "
                          f"Steps: {steps} | ε: {agent.epsilon:.3f} | "
                          f"Success: {100*success_count/(episode+1):.1f}%")

            except Exception as e:
                print(f"[{self.worker_id}] ERROR in episode {episode}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save final checkpoint
        checkpoint_path = f'checkpoints/{self.worker_id}_final_agent.pt'
        os.makedirs('checkpoints', exist_ok=True)
        agent.save(checkpoint_path)

        # Calculate final results
        results = {
            'worker_id': self.worker_id,
            'episodes': episodes,
            'query': query,
            'avg_reward': float(np.mean(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'avg_similarity': float(np.nanmean(episode_similarities)),
            'max_similarity': float(np.nanmax(episode_similarities)),
            'success_rate': 100 * success_count / episodes,
            'dead_end_rate': 100 * dead_end_count / episodes,
            'total_training_steps': total_training_steps,
            'checkpoint_path': checkpoint_path
        }

        print(f"\n{'='*80}")
        print(f"[{self.worker_id}] TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Episodes: {episodes}")
        print(f"Avg Reward: {results['avg_reward']:.2f}")
        print(f"Max Similarity: {results['max_similarity']:.3f}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n")

        logger.finish()

        return results
    

def rl_training_function(config):
    import os, sys
    print("CWD:", os.getcwd())
    print("PYTHONPATH:", sys.path)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    
    worker_id = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"[RayTrain Worker {worker_id}/{world_size}] Starting...")
    
    trainer = DistributedRLTrainer(
        worker_id=f"raytrain_w{worker_id}",
        db_host=config["db_host"],
        db_port=config["db_port"],
        db_user=config["db_user"],
        db_password=config["db_password"],
        db_name=config["db_name"]
    )

    results = asyncio.run(trainer.train_episodes(  
        episodes=config["episodes"],
        query=config["query"],
        start_paper_id=config["start_paper_id"],
        **config.get("training_config", {})
    ))
    
    train.report(results)

def main():
    with open("utils/config/cluster_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    db_config = config["database_server"]
    
    ray.init(address="auto" ,
            runtime_env={
                "working_dir": ".",
                "py_modules": ["RL", "graph", "distribute", "utils"],
                "env_vars": {
                    "HF_HOME": "/tmp/hf_cache",
                    "TRANSFORMERS_CACHE": "/tmp/hf_cache"
                }
            }
        )
    
    print(f"Ray cluster: {ray.cluster_resources()}")
    
    scaling_config = ScalingConfig(
        num_workers=3,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1},
        placement_strategy="SPREAD"
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=rl_training_function,
        train_loop_config={
            "episodes": 100,
            "query": "deep learning Stanford 2020-2023",
            "start_paper_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "db_host": db_config["host"],
            "db_port": db_config["neo4j_port"],
            "db_user": db_config["neo4j_user"],
            "db_password": db_config["neo4j_password"],
            "db_name": db_config["database_name"],
            "training_config": config.get("training_config", {})
        },
        scaling_config=scaling_config,
    )

    
    print("Starting 3-node distributed training...")
    result = trainer.fit()
    print("Training complete!")
    print(f"Results: {result.metrics}")
    
    ray.shutdown()

if __name__ == "__main__":
    main()
