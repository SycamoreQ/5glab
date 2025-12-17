import asyncio
import numpy as np
import pickle
import os
import time
from sentence_transformers import SentenceTransformer
from RL.ddqn import DDQLAgent
from RL.env import AdvancedGraphTraversalEnv, RelationType
from graph.database.store import EnhancedStore
from model.llm.parser.dspy_parser import DSPyHierarchicalParser, HierarchicalRewardMapper
import math
from collections import Counter

COMPLEX_QUERIES = [
    "deep learning transformers",
    "citations of Attention Is All You Need paper",
    "papers that cite BERT",
    "references of ResNet paper",
    "who are the authors of ImageNet paper",
    "recent papers by Geoffrey Hinton",
    "collaborators of Yann LeCun",
    "papers by Yoshua Bengio on deep learning",
    "papers on computer vision with more than 100 citations",
    "recent NLP papers published in ACL",
    "deep learning papers from Stanford 2020-2023",
    "second-order citations of GPT-3 paper",
    "Get me authors who wrote papers in 'IEEJ Transactions' in the field of Physics",
    "papers on transformers with more than 100 citations from 2020-2023",
    "collaborators of Yann LeCun who published in NeurIPS"
]

def normalize_paper_id(paper_id: str) -> str:
    """Normalize paper ID to consistent format."""
    if not paper_id:
        return ""
    # Convert to string and strip whitespace
    paper_id = str(paper_id).strip()
    # Remove leading zeros
    paper_id = paper_id.lstrip('0')
    # If removing zeros made it empty, it was "0000..." so return "0"
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
    print("Loading training cache...")
    
    cache_dir = 'training_cache'
    
    papers_file = os.path.join(cache_dir, 'training_papers_1M.pkl')
    with open(papers_file, 'rb') as f:
        papers = pickle.load(f)
    print(f"✓ Loaded {len(papers):,} papers")
    
    edges_file = os.path.join(cache_dir, 'edge_cache_1M.pkl')
    with open(edges_file, 'rb') as f:
        edge_cache = pickle.load(f)
    print(f"✓ Loaded edge cache with {sum(len(e) for e in edge_cache.values())//2:,} edges")
    
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
    embeddings_raw = encoder.cache 
    
    # Normalize all embedding keys
    embeddings = {}
    for k, v in embeddings_raw.items():
        normalized_k = normalize_paper_id(str(k))
        embeddings[normalized_k] = v
    
    # Normalize all edge cache IDs
    edge_cache_str = {}
    for src, edges in edge_cache.items():
        src_normalized = normalize_paper_id(str(src))
        normalized_edges = []
        for et, tid in edges:
            tid_normalized = normalize_paper_id(str(tid))
            normalized_edges.append((et, tid_normalized))
        edge_cache_str[src_normalized] = normalized_edges
    edge_cache = edge_cache_str

    print(f"✓ Loaded {len(embeddings):,} embeddings")
    
    return papers, edge_cache, paper_id_set, embeddings, encoder


async def prepare_query_pools(encoder, papers, paper_id_set):
    print("\nPreparing query pools...")
    
    query_pools = {}
    
    for query in COMPLEX_QUERIES:
        query_emb = encoder.encode([query])[0]
        
        scored_papers = []
        for paper in papers:
            pid = normalize_paper_id(str(paper.get('paperId') or paper.get('paper_id')))
            
            if pid not in paper_id_set:
                continue
            
            if pid not in encoder.cache:
                continue
            
            paper_emb = encoder.cache[pid]
            sim = np.dot(query_emb, paper_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
            )
            
            scored_papers.append((sim, paper))
        
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        query_pools[query] = [p for _, p in scored_papers[:100]] 
        
        print(f"  {query[:50]:50s} -> {len(query_pools[query]):3d} papers (max_sim: {scored_papers[0][0]:.3f})")
    
    return query_pools

async def train():
    papers, edge_cache, paper_id_set, embeddings, encoder = await build_embeddings()
    
    store = EnhancedStore(pool_size=10)

    env = AdvancedGraphTraversalEnv(
        store, 
        precomputed_embeddings=embeddings,
        use_communities=False,  
        use_feedback=False,
        use_llm_parser=True,  
        parser_type='dspy', 
        require_precomputed_embeddings=False
    )
    
    embedded_ids = set(embeddings.keys())
    
    # Normalize paper_id_set
    normalized_paper_id_set = {normalize_paper_id(str(pid)) for pid in paper_id_set}
    env.training_paper_ids = normalized_paper_id_set

    def norm_et(et: str) -> str:
        return et.lower().replace("_", "")
    
    pruned_edge_cache = {}
    for src, edges in edge_cache.items():
        src = normalize_paper_id(str(src))
        if src not in embedded_ids:
            continue
        kept = []
        for et, tid in edges:
            etn = norm_et(et)
            tid = normalize_paper_id(str(tid))
            if etn in ("cites", "citedby") and tid in embedded_ids:
                kept.append((etn, tid))
        if kept:
            pruned_edge_cache[src] = kept

    env.training_edge_cache = pruned_edge_cache
    env.precomputed_embeddings = embeddings

    print(f"\n✓ Graph statistics:")
    print(f"  Papers: {len(env.training_paper_ids):,}")
    print(f"  Edges: {sum(len(v) for v in env.training_edge_cache.values()):,}")
    
    avg_degree = sum(len(v) for v in env.training_edge_cache.values()) / len(env.training_paper_ids) if env.training_paper_ids else 0
    print(f"  Avg degree: {avg_degree:.1f}")

    # Diagnostic: Check ID overlap
    print("\n" + "="*80)
    print("CACHE ALIGNMENT CHECK")
    print("="*80)
    sample_papers = list(env.training_edge_cache.keys())[:20]
    for pid in sample_papers:
        edges = env.training_edge_cache[pid]
        has_embedding = pid in embeddings
        target_ids = [tid for _, tid in edges]
        targets_with_embeddings = sum(1 for tid in target_ids if tid in embeddings)
        print(f"Paper {pid[-8:]:>8s}: "
              f"edges={len(edges):2d} | "
              f"has_emb={'✓' if has_embedding else '✗'} | "
              f"targets_with_emb={targets_with_embeddings}/{len(target_ids)}")
    print("="*80 + "\n")

    query_pools = await prepare_query_pools(encoder, papers, normalized_paper_id_set)
    
    agent = DDQLAgent(
        state_dim=773, 
        text_dim=384, 
        use_prioritized=True,
        precomputed_embeddings=embeddings
    )

    c = Counter(et for edges in env.training_edge_cache.values() for et, _ in edges)
    print(f"Edge type distribution: {c}")

    import random
    if len(env.training_edge_cache) >= 100:
        pids = random.sample(list(env.training_edge_cache.keys()), 100)
    else:
        pids = list(env.training_edge_cache.keys())

    def has(pid, t): 
        return any(et == t for et, _ in env.training_edge_cache[pid])

    print(f"Papers with 'cites': {sum(has(pid,'cites') for pid in pids)}")
    print(f"Papers with 'citedby': {sum(has(pid,'citedby') for pid in pids)}")

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Queries: {len(COMPLEX_QUERIES)}")
    print(f"Max steps per episode: 12")
    print(f"Allow revisits: True (max 3x per node)")
    print("="*80 + "\n")
    
    episode_rewards = []
    episode_similarities = []
    episode_lengths = []
    dead_end_count = 0
    success_count = 0
    
    for episode in range(100):
        if episode < 50:
            env.max_revisits = 5
        else:
            env.max_revisits = 3

        query = np.random.choice(COMPLEX_QUERIES)
        paper_pool = query_pools[query]
        
        if not paper_pool:
            continue
        
        start_idx = np.random.choice(min(10, len(paper_pool)))
        start_paper_data = paper_pool[start_idx]
        if isinstance(start_paper_data, tuple):
            start_paper, start_paper_id, degree = start_paper_data
        else:
            start_paper = start_paper_data
            start_paper_id = normalize_paper_id(str(start_paper.get('paperId') or start_paper.get('paper_id')))

        start_pid = normalize_paper_id(str(start_paper.get('paperId') or start_paper.get('paper_id')))
        if start_pid not in env.training_edge_cache:
            continue

        neighbor_ids = [tid for _, tid in env.training_edge_cache[start_pid]]
        if not any(nid in embedded_ids for nid in neighbor_ids):
            continue
        
        edges = env.training_edge_cache.get(start_paper_id)
        if not edges:
            continue
        
        try:
            state = await env.reset(query, intent=1, start_node_id=start_paper_id)
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            continue

        edges_count = len(env.training_edge_cache.get(start_paper_id, []))
        print(f"[DEBUG] Starting paper {start_paper_id[-8:]}: {edges_count} edges in cache")
        
        episode_reward = 0
        steps = 0
        max_steps = 12
        hit_dead_end = False

        for step in range(max_steps):
            try:
                pid = normalize_paper_id(str(env.current_node.get("paperId") or 
                        env.current_node.get("paperid") or 
                        env.current_node.get("paper_id")))
                
                available = get_available_cache_relations(env, pid)
                if not available:
                    print(f"[DEBUG] No available relations at step {step}")
                    hit_dead_end = True
                    break

                relation_type = int(np.random.choice(available))
                is_terminal, manager_reward = await env.manager_step(relation_type)
                print(f"[DEBUG] Manager returned {len(env.available_worker_nodes)} worker actions")
                episode_reward += manager_reward
                
                if is_terminal:
                    break

                worker_actions = await env.get_worker_actions()
                print(f"[DEBUG] Sample action structure:")
                if worker_actions:
                    sample = worker_actions[0]
                    print(f"  Type: {type(sample)}")
                    print(f"  Length: {len(sample) if isinstance(sample, tuple) else 'N/A'}")
                    if isinstance(sample, tuple) and len(sample) == 2:
                        node, rel = sample
                        print(f"  Node type: {type(node)}")
                        print(f"  Node keys: {list(node.keys())[:5] if isinstance(node, dict) else 'N/A'}")
                        print(f"  Relation: {rel}")
                
                if not worker_actions:
                    print(f"[DEBUG] No worker actions available")
                    hit_dead_end = True
                    break
                
                # Filter for embedded nodes
                worker_actions = [
                    (n, r) for (n, r) in worker_actions
                    if normalize_paper_id(str(n.get("paperid") or n.get("paperId") or n.get("paper_id"))) in embedded_ids
                ]
                
                if not worker_actions:
                    print(f"[DEBUG] No valid worker actions after filtering")
                    hit_dead_end = True
                    break

                worker_actions = worker_actions[:5]
                
                valid_worker_actions = []
                for action in worker_actions:
                    if isinstance(action, tuple) and len(action) == 2:
                        node, rel = action
                        if isinstance(node, dict) and node:
                            valid_worker_actions.append(action)
                        else:
                            print(f"[DEBUG] Skipping action with invalid node: {type(node)}")
                    else:
                        print(f"[DEBUG] Skipping malformed action: {type(action)}")
                
                if not valid_worker_actions:
                    print(f"[DEBUG] No valid actions after validation")
                    hit_dead_end = True
                    break
                
                best_action = agent.act(state, valid_worker_actions, max_actions=5)
                
                if best_action is None:
                    print(f"[DEBUG] agent.act() returned None")
                    hit_dead_end = True
                    break
                
                if not isinstance(best_action, tuple) or len(best_action) != 2:
                    print(f"[DEBUG] agent.act() returned malformed action: {type(best_action)}")
                    hit_dead_end = True
                    break

                chosen_node, chosen_relation = best_action
                
                if chosen_node is None or not isinstance(chosen_node, dict):
                    print(f"[DEBUG] Chosen node is invalid: {type(chosen_node)}")
                    hit_dead_end = True
                    break
                
                # Proceed with worker step
                next_state, worker_reward, done = await env.worker_step(chosen_node)
                episode_reward += worker_reward
                steps += 1
                
                next_actions = await env.get_worker_actions() if not done else []
                next_actions = [
                    (n, r) for n, r in next_actions 
                    if normalize_paper_id(str(n.get('paperId') or n.get('paper_id'))) in embedded_ids
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
                import traceback
                print(f"[ERROR] Step {step} failed: {e}")
                print(traceback.format_exc())
                hit_dead_end = True
                break


        if steps < 2:
            dead_end_count += 1
        
        loss = 0.0
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
        
        if episode % 10 == 0 and episode > 0:
            agent.update_target()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        final_sim = env.best_similarity_so_far
        episode_similarities.append(final_sim if final_sim > -0.5 else np.nan)
        
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
            print(f"Episode {episode+1}/100")
            print(f"{'='*70}")
            print(f"  Query:          {query[:50]}")
            print(f"  Reward:       {episode_reward:+7.2f} | Avg: {avg_reward:+7.2f}")
            print(f"  Similarity:     {final_sim:.3f} | Avg: {avg_sim:.3f}")
            print(f"  Best Sim:       {best_sim_overall:.3f}")
            print(f"  Steps:          {steps:2d} | Avg: {avg_length:.1f}")
            print(f"  Success rate:   {success_rate:.1f}% (sim > 0.5)")
            print(f"  Dead ends:      {dead_end_rate:.1f}%")
            print(f"  Loss: {loss:.4f} | ε: {agent.epsilon:.3f}")
            
            if hasattr(env, 'current_query_facets') and env.current_query_facets:
                facets = env.current_query_facets
                print(f"  Parsed: {facets.get('target_entity', 'N/A')}/{facets.get('operation', 'N/A')} | {len(facets.get('constraints', []))} constraints")
            
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
