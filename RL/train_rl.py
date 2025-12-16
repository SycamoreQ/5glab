import asyncio
import numpy as np
import pickle
import os
import time
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
import math

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

def find_papers_to_embed():
    import pickle

    print("Loading graph...")
    with open('pruned_graph_1M.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    citations = graph_data['citations']
    papers = graph_data['papers']

    # Get ALL paper IDs in citations
    all_citation_ids = set()
    for cite in citations:
        all_citation_ids.add(str(cite['source']))
        all_citation_ids.add(str(cite['target']))

    print(f"Unique papers in citations: {len(all_citation_ids):,}")

    # Check current coverage
    with open('training_cache/embeddings_1M.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    embedded_ids = {str(k) for k in embeddings.keys()}
    missing_ids = all_citation_ids - embedded_ids

    print(f"Already embedded: {len(embedded_ids):,}")
    print(f"Missing embeddings: {len(missing_ids):,}")

    paper_dict = {str(p['paperId']): p for p in papers}
    missing_papers = [paper_dict[pid] for pid in missing_ids if pid in paper_dict]

    print(f"Papers available to embed: {len(missing_papers):,}")

    # Save for embedding
    with open('training_cache/papers_to_embed.pkl', 'wb') as f:
        pickle.dump(missing_papers, f)

    print("\n✓ Saved papers_to_embed.pkl")




async def build_query_pools_from_cache(papers, embeddings, encoder, queries, top_k=200):
    """Build query-specific paper pools using semantic similarity."""
    print("\nBuilding query-specific paper pools...")
    
    query_embeddings = encoder.encode(queries)
    query_pools = {}
    
    for query, query_emb in zip(queries, query_embeddings):
        similarities = []

        sample_size = min(50000, len(papers))
        sampled_papers = papers if len(papers) <= sample_size else papers[:sample_size]
        
        for paper in sampled_papers:
            # Handle both key formats
            pid = paper.get('paperId') or paper.get('paper_id')
            if pid and pid in embeddings:
                paper_emb = embeddings[pid]
                sim = np.dot(query_emb, paper_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
                )
                similarities.append((sim, paper))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        query_pools[query] = [paper for _, paper in similarities[:top_k]]
        
        print(f"  '{query[:40]}...': {len(query_pools[query])} papers")
        if similarities:
            print(f"    Top similarity: {similarities[0][0]:.3f}")
    
    return query_pools

async def train():
    """Main training loop."""
    
    papers, edge_cache, paper_id_set = await load_training_cache()
    
    store = EnhancedStore(pool_size=10)

    print("Loading missing papers...")
    with open('training_cache/papers_to_embed.pkl', 'rb') as f:
        missing_papers = pickle.load(f)

    print(f"Embedding {len(missing_papers):,} papers...")
    
    from utils.batchencoder import BatchEncoder
    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'
    )
    encoder.precompute_paper_embeddings(missing_papers , force = False )
    
    # Precompute embeddings
    embeddings_dict = encoder.precompute_paper_embeddings(papers, force=False)
    embeddings = encoder.cache 
    embeddings = {str(k): v for k, v in embeddings.items()}

    edge_cache_str = {}
    for src, edges in edge_cache.items():
        src = str(src)
        edge_cache_str[src] = [(et, str(tid)) for et, tid in edges]
    edge_cache = edge_cache_str
    
    print(f"Embeddings ready: {len(embeddings):,}")
    queries = [
    # broad queries 
        "deep learning convolutional neural networks",
        "natural language processing transformers",
        "reinforcement learning policy gradient",
        
        # Specific architecture queries
        "ResNet deep residual learning image recognition",
        "BERT bidirectional encoder representations transformers",
        "proximal policy optimization PPO reinforcement learning",
        
        # Very specific queries 
        "attention is all you need vaswani transformer",
        "deep double Q-network DDQN prioritized replay",
        "AlexNet ImageNet convolutional neural network"
    ]

    
    query_pools = None
    try:
        with open('training_cache/query_pools_1M.pkl', 'rb') as f:
            cached_pools = pickle.load(f)
        
        if all(q in cached_pools for q in queries):
            query_pools = {q: cached_pools[q] for q in queries}
            print(f"✓ Loaded pre-built query pools for {len(queries)} queries")
            
            for q, pool in query_pools.items():
                print(f"  '{q[:40]}...': {len(pool)} papers")
                if not pool:
                    print(f"    ⚠ Empty pool, will rebuild")
                    query_pools = None
                    break
        else:
            print("⚠ Cached pools don't match queries, rebuilding...")
            query_pools = None
    
    except FileNotFoundError:
        print("⚠ Query pools not found, building from scratch...")
        query_pools = None
    
    if query_pools is None:
        query_pools = await build_query_pools_from_cache(papers, embeddings, encoder, queries)
        
        with open('training_cache/query_pools_1M.pkl', 'wb') as f:
            pickle.dump(query_pools, f)
        print("✓ Saved query pools")
    
    if not query_pools or all(not pool for pool in query_pools.values()):
        raise ValueError("All query pools are empty! Cannot train.")
    
    queries = [q for q in queries if q in query_pools and query_pools[q]]
    if not queries:
        raise ValueError("No valid queries with papers!")
    
    print(f"✓ Using {len(queries)} queries for training\n")
    
    # Create environment
    env = AdvancedGraphTraversalEnv(
        store, 
        precomputed_embeddings=embeddings,
        use_communities=False, 
        use_feedback=False,
        require_precomputed_embeddings=True
    )

    # Build dense subgraph
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


    print(f"\nGraph statistics:")
    print(f"  Papers: {len(env.training_paper_ids):,}")
    print(f"  Edges: {sum(len(v) for v in env.training_edge_cache.values()):,}")

    # Degree distribution
    out_degree = {src: len(edges) for src, edges in env.training_edge_cache.items()}
    from collections import Counter
    out_counts = Counter(out_degree.values())
    print(f"\n📈 Degree distribution (top 10):")
    for d in sorted(out_counts.keys())[:10]:
        print(f"  Degree {d}: {out_counts[d]:,} papers")

    # Connected components 
    visited_bfs = set()
    components = []

    def bfs(start):
        queue = [start]
        component = set()
        while queue:
            node = queue.pop(0)
            if node in visited_bfs:
                continue
            visited_bfs.add(node)
            component.add(node)
            if node in env.training_edge_cache:
                for _, neighbor in env.training_edge_cache[node]:
                    if neighbor not in visited_bfs:
                        queue.append(neighbor)
        return component

    for pid in list(env.training_paper_ids)[:2000]:  # sample 2000
        if pid not in visited_bfs:
            comp = bfs(pid)
            if len(comp) > 1:
                components.append(len(comp))

    if components:
        components.sort(reverse=True)
        print(f"\nConnected components (top 5):")
        for i, size in enumerate(components[:5]):
            print(f"  Component {i+1}: {size:,} papers")
    print("\n" + "="*80)

    print(" MAXIMUM POSSIBLE SIMILARITY DIAGNOSTIC")
    print("="*80)

    # Build paper titles dictionary from your papers list
    paper_titles = {}
    for p in papers:
        pid = str(p.get('paperId') or p.get('paper_id'))
        title = p.get('title') or 'Unknown Title' 
        paper_titles[pid] = title

    print(f"Built title index for {len(paper_titles):,} papers\n")

    # Check max similarity for each query
    for query in queries:
        print(f"\n Query: '{query}'")
        print("-" * 80)
        
        # Encode query
        query_emb = encoder.encode([query])[0]

        all_sims = []
        for pid, paper_emb in embeddings.items():
            sim = np.dot(query_emb, paper_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
            )
            all_sims.append((sim, pid))

        all_sims.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Top 10 most similar papers in entire corpus:")
        for rank, (sim, pid) in enumerate(all_sims[:10], 1):
            title = paper_titles.get(pid, 'Unknown')[:70]
            in_graph = "✓" if pid in paper_id_set else "✗"
            print(f"  {rank:2d}. {sim:.3f} [{in_graph}] {title}")
        
        top_10_avg = np.mean([s for s, _ in all_sims[:10]])
        top_100_avg = np.mean([s for s, _ in all_sims[:100]])
        max_sim = all_sims[0][0]
        
        print(f"\n  Max possible:     {max_sim:.3f}")
        print(f"  Top-10 average:   {top_10_avg:.3f}")
        print(f"  Top-100 average:  {top_100_avg:.3f}")
        
        top_5_in_graph = sum(1 for _, pid in all_sims[:5] if pid in paper_id_set)
        top_20_in_graph = sum(1 for _, pid in all_sims[:20] if pid in paper_id_set)
        
        print(f"  Top-5 in graph:   {top_5_in_graph}/5")
        print(f"  Top-20 in graph:  {top_20_in_graph}/20")

    print("\n" + "="*80 + "\n")

    

    for q in list(query_pools.keys()):
        pool = query_pools[q]
        if not pool:
            continue
        
        similarities = []
        q_emb = encoder.encode([q])[0]
        
        for p in pool:
            pid = str(p.get('paperId') or p.get('paper_id'))
            if pid in embeddings:
                p_emb = embeddings[pid]
                sim = np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-9)
                similarities.append((sim, p))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        query_pools[q] = [p for _, p in similarities]
        
        print(f"\nQuery: {q[:50]}...")
        for i, (sim, p) in enumerate(similarities[:5]):
            title = p.get('title', 'N/A')[:60]
            print(f"  {i+1}. {sim:.3f} - {title}")

    print()

    some_src = next(iter(env.training_edge_cache))
    print("Sample edges:", env.training_edge_cache[some_src][:5])

    print(f"✓ Set training_paper_ids (dense only): {len(env.training_paper_ids):,} papers")
    print(f"✓ Pruned edge cache: {sum(len(v) for v in env.training_edge_cache.values()):,} edges")
    
    # Initialize agent
    agent = DDQLAgent(
        state_dim=773, 
        text_dim=384, 
        use_prioritized=True,
        precomputed_embeddings=embeddings
    )
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Training papers: {len(papers):,}")
    print(f"Embeddings: {len(embeddings):,}")
    print(f"Dense subgraph: {len(env.training_paper_ids):,} papers")
    print(f"Edge cache: {sum(len(e) for e in env.training_edge_cache.values()):,} edges")
    print(f"Queries: {len(queries)}")
    print(f"Max steps per episode: 5")
    print()
    
    # Training loop
    episode_rewards = []
    episode_similarities = []
    episode_lengths = []
    dead_end_count = 0
    success_count = 0
    
    for episode in range(1000):  
        query = np.random.choice(queries)
        paper_pool = query_pools[query]
        
        if not paper_pool:
            print(f"[SKIP] No papers for query: {query}")
            continue
        
        # Start from top-10 most similar papers 
        start_idx = np.random.choice(min(10, len(paper_pool)))
        
        start_paper = paper_pool[start_idx]
        
        start_paper_id = start_paper.get('paperId') or start_paper.get('paper_id')
        edges = env.training_edge_cache.get(str(start_paper_id), None) or env.training_edge_cache.get(start_paper_id, None)
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
                    pid = str(env.current_node.get("paperId") or env.current_node.get("paperid") or env.current_node.get("paper_id"))
                    edges = env.training_edge_cache.get(pid, [])

                    has_cites = any(et == "cites" for et, _ in edges)
                    has_cited_by = any(et in ("cited_by", "citedby") for et, _ in edges)

                    manageractions = []
                    if has_cites: 
                        manageractions.append(RelationType.CITES)
                    if has_cited_by: 
                        manageractions.append(RelationType.CITED_BY)

                    if not manageractions:
                        done = True
                        break

                    relationtype = int(np.random.choice(manageractions))
                    isterminal, managerreward = await env.manager_step(relationtype)
                    episode_reward += managerreward
                    if isterminal:
                        done = True
                        break

                    worker_actions = await env.get_worker_actions()
                    #worker_actions = [
                    #    (n, r) for (n, r) in worker_actions
                    #    if (str(n.get("paperid") or n.get("paperId") or n.get("paper_id")) in env.training_paper_ids)
                    #]
                    if worker_actions:
                        break  

                if done or not worker_actions:
                    break

                worker_actions = worker_actions[:5]
                best_action = agent.act(state, worker_actions, max_actions=5)
                if not best_action:
                    break

                chosen_node, _ = best_action
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                steps += 1
                
                # Store experience
                next_actions = await env.get_worker_actions() if not done else []
                next_actions = [
                    (n, r) for n, r in next_actions 
                    if str(n.get('paperId') or n.get('paper_id')) in env.training_paper_ids
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
        
        # Training
        loss = 0.0
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
        
        # Update target network
        if episode % 10 == 0 and episode > 0:
            agent.update_target()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        final_sim = env.best_similarity_so_far
        episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)
        
        if hit_dead_end:
            dead_end_count += 1
        
        if final_sim > 0.5:
            success_count += 1
        
        # Logging every 10 episodes
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
            print(f"  Reward:       {episode_reward:+7.2f} | Avg: {avg_reward:+7.2f}")
            print(f"  Similarity:     {final_sim:.3f} | Avg: {avg_sim:.3f}")
            print(f"  Best Sim:       {best_sim_overall:.3f}")
            print(f"  Steps:          {steps:2d} | Avg: {avg_length:.1f}")
            print(f"  Success rate:   {success_rate:.1f}% (sim > 0.5)")
            print(f"  Dead ends:      {dead_end_rate:.1f}%")
            print(f"  Loss: {loss:.4f} | ε: {agent.epsilon:.3f}")
            print(f"{'='*70}\n")
        
        # Save checkpoints
        if (episode + 1) % 100 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(f'checkpoints/agent_1M_ep{episode+1}.pt')
            print(f"[CHECKPOINT] Saved at episode {episode+1}")
    
    # Final save
    agent.save('final_agent_1M.pt')
    
    # Save stats
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
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes:     {len(episode_rewards)}")
    print(f"Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Success rate:       {stats['success_rate']:.1f}%")
    print(f"Dead end rate:      {stats['dead_end_rate']:.1f}%")
    if episode_similarities:
        print(f"Best similarity:    {float(np.nanmax(episode_similarities)):.3f}")
        print(f"Avg similarity:     {float(np.nanmean(episode_similarities)):.3f}")
    
    await store.pool.close()

if __name__ == '__main__':
    asyncio.run(train())
