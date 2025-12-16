import asyncio
import numpy as np
import pickle
import os
import time
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
from model.llm.parser.unified import UnifiedQueryParser, ParserType  
import math

COMPLEX_QUERIES = [
    # Semantic
    "deep learning transformers",
    
    # Paper operations
    "citations of Attention Is All You Need paper",
    "papers that cite BERT",
    "references of ResNet paper",
    "who are the authors of ImageNet paper",
    
    # Author operations
    "recent papers by Geoffrey Hinton",
    "collaborators of Yann LeCun",
    "papers by Yoshua Bengio on deep learning",
    
    # Constraints
    "papers on computer vision with more than 100 citations",
    "recent NLP papers published in ACL",
    "deep learning papers from Stanford 2020-2023",
    
    # Multi-hop
    "second-order citations of GPT-3 paper",
]

async def load_training_cache():
    """Load the training cache."""
    print("Loading training cache...")
    
    cache_dir = 'training_cache'
    
    # Load papers
    papers_file = os.path.join(cache_dir, 'training_papers_1M.pkl')
    with open(papers_file, 'rb') as f:
        papers = pickle.load(f)
    print(f"✓ Loaded {len(papers):,} papers")
    
    # Load edge cache
    edges_file = os.path.join(cache_dir, 'edge_cache_1M.pkl')
    with open(edges_file, 'rb') as f:
        edge_cache = pickle.load(f)
    print(f"✓ Loaded edge cache with {sum(len(e) for e in edge_cache.values())//2:,} edges")
    
    # Load paper ID set
    index_file = os.path.join(cache_dir, 'paper_id_set_1M.pkl')
    with open(index_file, 'rb') as f:
        paper_id_set = pickle.load(f)
    print(f"✓ Loaded paper ID index with {len(paper_id_set):,} IDs")
    
    return papers, edge_cache, paper_id_set

async def build_embeddings():
    papers, edge_cache, paper_id_set = await load_training_cache()

    print("Loading/building embeddings...")
    
    from utils.batchencoder import BatchEncoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'
    )
    
    embeddings_dict = encoder.precompute_paper_embeddings(papers, force=False)
    embeddings = encoder.cache 
    embeddings = {str(k): v for k, v in embeddings.items()}


    edge_cache_str = {}
    for src, edges in edge_cache.items():
        src = str(src)
        edge_cache_str[src] = [(et, str(tid)) for et, tid in edges]
    edge_cache = edge_cache_str

    print(f"✓ Loaded {len(embeddings):,} embeddings")
    
    return papers, edge_cache, paper_id_set, embeddings, encoder


async def prepare_query_pools(encoder, papers, paper_id_set):
    """Pre-compute query pools for faster training."""
    print("\nPreparing query pools...")
    
    query_pools = {}
    
    for query in COMPLEX_QUERIES:
        # Encode query
        query_emb = encoder.encode([query])[0]
        
        # Score all papers
        scored_papers = []
        for paper in papers:
            pid = str(paper.get('paperId') or paper.get('paper_id'))
            
            # Skip if not in graph
            if pid not in paper_id_set:
                continue
            
            # Skip if no embedding
            if pid not in encoder.cache:
                continue
            
            paper_emb = encoder.cache[pid]
            sim = np.dot(query_emb, paper_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
            )
            
            scored_papers.append((sim, paper))
        
        # Sort by similarity
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Store top papers
        query_pools[query] = [p for _, p in scored_papers[:100]] 
        
        print(f"  {query[:50]:50s} -> {len(query_pools[query]):3d} papers (max_sim: {scored_papers[0][0]:.3f})")
    
    return query_pools


async def train():
    """Main training loop."""
    query = np.random.choice(COMPLEX_QUERIES)
    
    papers, edge_cache, paper_id_set, embeddings, encoder = await build_embeddings()
    
    store = EnhancedStore(pool_size=10)


    query_parser = UnifiedQueryParser(
        primary_parser=ParserType.RULE_BASED, 
        fallback_on_error=True
    )
    print("✓ Loaded query parser")

    env = AdvancedGraphTraversalEnv(
        store, 
        precomputed_embeddings=embeddings,
        use_communities=False,  
        use_feedback=False,
        require_precomputed_embeddings=True
    )
    embedded_ids = set(embeddings.keys())
    env.training_paper_ids = embedded_ids  
    
    pruned_edge_cache = {}
    for src, edges in edge_cache.items():
        if src not in embedded_ids:
            continue
        kept = [(et, tid) for (et, tid) in edges if tid in embedded_ids]
        if kept:
            pruned_edge_cache[src] = kept

    env.training_edge_cache = pruned_edge_cache

    print(f"\n✓ Graph statistics:")
    print(f"  Papers: {len(env.training_paper_ids):,}")
    print(f"  Edges: {sum(len(v) for v in env.training_edge_cache.values()):,}")

    query_pools = await prepare_query_pools(encoder, papers, paper_id_set)
    
    print("\n" + "="*80)
    print(" MAXIMUM POSSIBLE SIMILARITY DIAGNOSTIC")
    print("="*80)

    paper_titles = {}
    for p in papers:
        pid = str(p.get('paperId') or p.get('paper_id'))
        title = p.get('title') or 'Unknown Title' 
        paper_titles[pid] = title

    for query in QUERIES[:3]:  # Check first 3 queries
        print(f"\n📊 Query: '{query}'")
        print("-" * 80)
        
        query_emb = encoder.encode([query])[0]

        all_sims = []
        for pid, paper_emb in embeddings.items():
            sim = np.dot(query_emb, paper_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
            )
            all_sims.append((sim, pid))

        all_sims.sort(key=lambda x: x[0], reverse=True)
        
        print(f"  Top 5 papers:")
        for rank, (sim, pid) in enumerate(all_sims[:5], 1):
            title = paper_titles.get(pid, 'Unknown')[:60]
            in_graph = "✓" if pid in paper_id_set else "✗"
            print(f"    {rank}. [{sim:.3f}] {in_graph} {title}")
        
        max_sim = all_sims[0][0]
        top_5_in_graph = sum(1 for _, pid in all_sims[:5] if pid in paper_id_set)
        
        print(f"  Max possible: {max_sim:.3f} | Top-5 in graph: {top_5_in_graph}/5")

    print("\n" + "="*80 + "\n")

    agent = DDQLAgent(
        state_dim=773, 
        text_dim=384, 
        use_prioritized=True,
        precomputed_embeddings=embeddings
    )
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    print(f"Training papers: {len(papers):,}")
    print(f"Embeddings: {len(embeddings):,}")
    print(f"Dense subgraph: {len(env.training_paper_ids):,} papers")
    print(f"Edge cache: {sum(len(e) for e in env.training_edge_cache.values()):,} edges")
    print(f"Queries: {len(QUERIES)}")
    print(f"Max steps per episode: 8")
    print(f"Target episodes: 500")
    print("="*80 + "\n")
    
    episode_rewards = []
    episode_similarities = []
    episode_lengths = []
    dead_end_count = 0
    success_count = 0
    
    for episode in range(500):  
        query = np.random.choice(QUERIES)
        paper_pool = query_pools[query]
        
        if not paper_pool:
            print(f"[SKIP] No papers for query: {query}")
            continue
        
        query_facets = query_parser.parse(query)
        query_mode = 'semantic' 
        

        if query_facets.get('author_search_mode'):
            query_mode = 'author'
        elif query_facets.get('paper_search_mode'):
            query_mode = 'paper'
        

        start_idx = np.random.choice(min(10, len(paper_pool)))
        start_paper = paper_pool[start_idx]
        start_paper_id = str(start_paper.get('paperId') or start_paper.get('paper_id'))
        
        edges = env.training_edge_cache.get(start_paper_id)
        if not edges:
            continue
        
        try:
            if hasattr(env, 'reset_with_cache_validation'):
                state = await env.reset_with_cache_validation(
                    query, 
                    intent=1, 
                    start_node_id=start_paper_id
                )
            else:
                state = await env.reset(
                    query, 
                    intent=1, 
                    start_node_id=start_paper_id
                )
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            continue
        
        episode_reward = 0
        steps = 0
        max_steps = 8
        hit_dead_end = False
        
        try:
            for step in range(max_steps):
                done = False

                for _attempt in range(3):
                    pid = str(env.current_node.get("paperId") or 
                             env.current_node.get("paperid") or 
                             env.current_node.get("paper_id"))
                    edges = env.training_edge_cache.get(pid, [])

                    has_cites = any(et == "cites" for et, _ in edges)
                    has_cited_by = any(et in ("cited_by", "citedby") for et, _ in edges)

                    manager_actions = []
                    if has_cites: 
                        manager_actions.append(RelationType.CITES)
                    if has_cited_by: 
                        manager_actions.append(RelationType.CITED_BY)

                    if not manager_actions:
                        done = True
                        break

                    # Manager step
                    relation_type = int(np.random.choice(manager_actions))
                    is_terminal, manager_reward = await env.manager_step(relation_type)
                    episode_reward += manager_reward
                    
                    if is_terminal:
                        done = True
                        break

                    # Get worker actions
                    worker_actions = await env.get_worker_actions()
                    
                    # Filter to only embedded papers
                    worker_actions = [
                        (n, r) for (n, r) in worker_actions
                        if str(n.get("paperid") or n.get("paperId") or n.get("paper_id")) in embedded_ids
                    ]
                    
                    if worker_actions:
                        break

                if done or not worker_actions:
                    break

                # Limit actions
                worker_actions = worker_actions[:5]
                
                # Agent selects action
                best_action = agent.act(state, worker_actions, max_actions=5)
                if not best_action:
                    break

                # Execute worker step
                chosen_node, _ = best_action
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                steps += 1
                
                # Store experience
                next_actions = await env.get_worker_actions() if not done else []
                next_actions = [
                    (n, r) for n, r in next_actions 
                    if str(n.get('paperId') or n.get('paper_id')) in embedded_ids
                ][:5]
                
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
                
        except Exception as e:
            print(f"  [ERROR] Step {step} failed: {e}")
            hit_dead_end = True
        
        loss = 0.0
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
        
        # Update target network
        if episode % 10 == 0 and episode > 0:
            agent.update_target()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        final_sim = env.best_similarity_so_far
        episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)
        
        if hit_dead_end:
            dead_end_count += 1
        
        if final_sim > 0.5:
            success_count += 1

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_sim = float(np.nanmean(episode_similarities[-50:])) if len(episode_similarities) >= 50 else float(np.nanmean(episode_similarities))
            best_sim_overall = float(np.nanmax(episode_similarities)) if len(episode_similarities) else float("nan")
            avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            success_rate = 100 * success_count / (episode + 1)
            dead_end_rate = 100 * dead_end_count / (episode + 1)
            
            print(f"\n{'='*70}")
            print(f"Episode {episode+1}/500")
            print(f"{'='*70}")
            print(f"  Query:          {query[:50]}")
            print(f"  Reward:       {episode_reward:+7.2f} | Avg: {avg_reward:+7.2f}")
            print(f"  Similarity:     {final_sim:.3f} | Avg: {avg_sim:.3f}")
            print(f"  Best Sim:       {best_sim_overall:.3f}")
            print(f"  Steps:          {steps:2d} | Avg: {avg_length:.1f}")
            print(f"  Success rate:   {success_rate:.1f}% (sim > 0.5)")
            print(f"  Dead ends:      {dead_end_rate:.1f}%")
            print(f"  Loss: {loss:.4f} | ε: {agent.epsilon:.3f}")
            print(f"{'='*70}\n")
        
        if (episode + 1) % 100 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(f'checkpoints/agent_1M_ep{episode+1}.pt')
            print(f"[CHECKPOINT] Saved at episode {episode+1}")
    
    agent.save('final_agent_1M.pt')
    
    stats = {
        'episode_rewards': episode_rewards,
        'episode_similarities': episode_similarities,
        'episode_lengths': episode_lengths,
        'success_rate': 100 * success_count / len(episode_rewards) if episode_rewards else 0,
        'dead_end_rate': 100 * dead_end_count / len(episode_rewards) if episode_rewards else 0
    }
    with open('training_stats_1M.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes:     {len(episode_rewards)}")
    print(f"Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Success rate:       {stats['success_rate']:.1f}%")
    print(f"Dead end rate:      {stats['dead_end_rate']:.1f}%")
    if episode_similarities:
        print(f"Best similarity:    {float(np.nanmax(episode_similarities)):.3f}")
        print(f"Avg similarity:     {float(np.nanmean(episode_similarities)):.3f}")
    print("="*80)
    
    await store.pool.close()

if __name__ == '__main__':
    asyncio.run(train())
