import asyncio
from graph.database.store import EnhancedStore
from RL.env import AdvancedGraphTraversalEnv, RelationType
import random 

async def test_community_rewards():
    print("=" * 80)
    print("COMMUNITY-AWARE REWARD SYSTEM TEST")
    print("=" * 80)
    
    store = EnhancedStore()
    env = AdvancedGraphTraversalEnv(store, use_communities=True)
    
    if not env.use_communities:
        print("\n⚠ Communities not available!")
        return
    
    print(f"✓ Communities loaded: {len(env.community_detector.communities)} nodes in cache")
    stats = env.community_detector.get_statistics()
    print(f"  Number of communities: {stats['num_communities']}")
    print(f"  Coverage: {stats['num_nodes']} nodes")
    
    print("\n1. Fetching well-connected paper from cache...")
    
    cached_paper_ids = list(env.community_detector.communities.keys())
    
    if not cached_paper_ids:
        print("✗ No papers in cache!")
        return
    
    paper = await store.get_well_connected_paper()
    
    if not paper:
        random_id = random.choice(cached_paper_ids)
        paper = await store.get_paper_by_id(random_id)
        print("  ⚠ Using random cached paper (no well-connected ones found)")
    
    if not paper:
        print("✗ No papers found!")
        return
    
    paper_id = paper.get('paper_id')
    query_text = paper.get('title') or paper.get('original_id') or 'research'
    
    print(f"✓ Starting paper: {query_text[:60]}...")
    print(f"  Paper ID: {paper_id}")
    
    # Verify it's in cache
    start_comm = env.community_detector.get_community(paper_id)
    if start_comm:
        print(f"  ✓ Paper IS in cache with community: {start_comm}")
        comm_size = env.community_detector.get_community_size(start_comm)
        print(f"  ✓ Community size: {comm_size} nodes")
    else:
        print(f"  ✗ ERROR: Paper should be in cache but isn't!")
        print(f"     This shouldn't happen. Please report this bug.")
        return
    
    # Reset
    state = await env.reset(query_text, RelationType.CITED_BY, start_node_id=paper_id)
    
    print(f"\n2. Initial community after reset: {env.current_community}")
    if env.use_communities and env.current_community:
        comm_size = env.community_detector.get_community_size(env.current_community)
        print(f"   Community size: {comm_size} nodes")
    elif env.current_community is None:
        print(f"   ⚠ Starting node not in any community (not in cache)")
    
    # Run episode
    print("\n3. Running episode with community tracking...\n")
    print("=" * 80)
    
    total_reward = 0.0
    step = 0
    done = False
    
    nodes_checked = 0
    nodes_with_community = 0
    
    while not done and step < 5:
        step += 1
        print(f"\nSTEP {step}")
        print("-" * 80)
        
        # Manager
        manager_actions = await env.get_manager_actions()
        if not manager_actions or (len(manager_actions) == 1 and RelationType.STOP in manager_actions):
            break
        
        if RelationType.STOP in manager_actions and len(manager_actions) > 1 and step < 4:
            manager_actions.remove(RelationType.STOP)
        
        manager_action = random.choice(manager_actions)
        is_terminal, manager_reward = await env.manager_step(manager_action)
        
        print(f"Manager reward: {manager_reward:+.4f}")
        total_reward += manager_reward
        
        if is_terminal:
            break
        
        # Worker
        worker_actions = await env.get_worker_actions()
        if not worker_actions:
            print("No worker actions available")
            continue
        
        print(f"Worker actions available: {len(worker_actions)}")
        
        # Check how many worker actions have communities
        actions_with_comm = 0
        for node, _ in worker_actions[:min(5, len(worker_actions))]:
            node_id = node.get('paper_id') or node.get('author_id')
            if node_id:
                nodes_checked += 1
                if env.community_detector.get_community(node_id):
                    actions_with_comm += 1
                    nodes_with_community += 1
        
        if actions_with_comm > 0:
            print(f"  ✓ {actions_with_comm}/{min(5, len(worker_actions))} checked nodes have communities")
        else:
            print(f"  ⚠ 0/{min(5, len(worker_actions))} checked nodes have communities (not in cache)")
        
        chosen_node, _ = random.choice(worker_actions)
        node_text = (
            chosen_node.get('title') or 
            chosen_node.get('name') or 
            'Unknown'
        )[:50]
        
        chosen_id = chosen_node.get('paper_id') or chosen_node.get('author_id')
        chosen_comm = env.community_detector.get_community(chosen_id) if chosen_id else None
        
        print(f"Chose: {node_text}...")
        print(f"  Node ID: {chosen_id}")
        print(f"  Pre-step community lookup: {chosen_comm if chosen_comm else '❌ NOT IN CACHE'}")
        
        # Track community before step
        prev_community = env.current_community
        steps_in_comm_before = env.steps_in_current_community
        
        next_state, worker_reward, done = await env.worker_step(chosen_node)
        
        # Show community change
        new_community = env.current_community
        
        print(f"\nWorker reward: {worker_reward:+.4f}")
        
        # Show community tracking
        if env.use_communities:
            if new_community and new_community != prev_community:
                print(f"  ✓ COMMUNITY SWITCH: {prev_community} → {new_community}")
                print(f"    (Was in {prev_community} for {steps_in_comm_before} steps)")
            elif new_community:
                print(f"  → Staying in community: {new_community}")
                print(f"    (Now {env.steps_in_current_community} steps in this community)")
                
                if env.steps_in_current_community >= env.config.STUCK_THRESHOLD:
                    print(f"    ⚠ STUCK WARNING! ({env.steps_in_current_community} steps)")
            else:
                print(f"  ⚠ Still no community (node not in cache)")
        
        # Show trajectory info
        traj = env.trajectory_history[-1]
        print(f"  Semantic similarity: {traj['similarity']:.4f}")
        if env.use_communities and traj['community_reward'] != 0:
            print(f"  Community reward: {traj['community_reward']:+.4f} ({traj['community_reason']})")
        
        total_reward += worker_reward
        state = next_state
    
    # Cache coverage analysis
    print("\n" + "=" * 80)
    print("CACHE COVERAGE ANALYSIS")
    print("=" * 80)
    
    if nodes_checked > 0:
        coverage_pct = (nodes_with_community / nodes_checked) * 100
        print(f"Nodes checked: {nodes_checked}")
        print(f"Nodes with communities: {nodes_with_community}")
        print(f"Coverage rate: {coverage_pct:.1f}%")
        
        if coverage_pct < 20:
            print(f"\n PROBLEM: Very low coverage ({coverage_pct:.1f}%)")
            print(f"   Most nodes are not in the community cache.")
            print(f"\n💡 SOLUTION:")
            print(f"   Rebuild cache with more nodes:")
            print(f"   rm communities.pkl")
            print(f"   python -m RL.community_detection")
        elif coverage_pct < 50:
            print(f"\n⚠ WARNING: Low coverage ({coverage_pct:.1f}%)")
            print(f"   Consider rebuilding cache with max_nodes=100000")
    
    print("\n" + "=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)
    
    summary = env.get_episode_summary()
    
    print(f"\nTotal Reward: {total_reward:+.4f}")
    print(f"Steps taken: {step}")
    
    if env.use_communities:
        print(f"\nCommunity Statistics:")
        print(f"  Unique communities visited: {summary['unique_communities_visited']}")
        print(f"  Community switches: {summary['community_switches']}")
        print(f"  Max steps in one community: {summary['max_steps_in_community']}")
        print(f"  Community loops: {summary['community_loops']}")
        print(f"  Community diversity ratio: {summary['community_diversity_ratio']:.2%}")
    
    print(f"\nOther Statistics:")
    print(f"  Unique relations: {summary['unique_relation_types']}")
    print(f"  Dead ends: {summary['dead_ends_hit']}")
    print(f"  Revisits: {summary['revisits']}")
    print(f"  Max similarity: {summary['max_similarity_achieved']:.4f}")
    
    # Show trajectory with communities
    print(f"\n" + "=" * 80)
    print("TRAJECTORY (with communities)")
    print("=" * 80)
    
    print(f"\n{'Step':<6} {'Node':<35} {'Comm':<15} {'Sim':<6} {'Reward':<8} {'Comm Reward'}")
    print("-" * 90)
    
    for i, traj in enumerate(env.trajectory_history, 1):
        node = traj['node']
        node_name = (node.get('title') or node.get('name') or 'Unknown')[:30]
        comm = str(traj['community'])[:12] if traj['community'] else "N/A"
        sim = traj['similarity']
        reward = traj['reward']
        comm_reward = traj['community_reward']
        
        print(f"{i:<6} {node_name:<35} {comm:<15} {sim:<6.3f} {reward:+8.3f} {comm_reward:+.3f}")
    
    # Evaluation
    print(f"\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    if summary['unique_communities_visited'] == 0:
        print("❌ No communities tracked - nodes not in cache")
        print("\n💡 Run diagnosis:")
        print("   python diagnose_community_issue.py")
    elif summary['community_switches'] >= 2:
        print("✓ Good exploration diversity (multiple communities)")
    else:
        print("⚠ Low exploration diversity (stuck in local area)")
    
    if summary['max_steps_in_community'] <= 3:
        print("✓ Avoiding getting stuck")
    elif summary['max_steps_in_community'] > 0:
        print(f"⚠ Got stuck in one community for {summary['max_steps_in_community']} steps")
    
    if summary['community_loops'] == 0:
        print("✓ No community loops (efficient exploration)")
    elif summary['community_loops'] > 0:
        print(f"⚠ Detected {summary['community_loops']} community loops")
    
    if total_reward > 3.0 and summary['unique_communities_visited'] > 0:
        print("\n✓✓✓ EXCELLENT! Community-aware rewards working well! ✓✓✓")
    elif total_reward > 0:
        print("\n✓ Good - positive rewards, but communities may not be tracked")
    else:
        print("\n⚠ Low rewards - agent getting stuck")
    
    await store.pool.close()


async def visualize_community_structure():
    """Visualize communities in the graph."""
    print("=" * 80)
    print("COMMUNITY STRUCTURE VISUALIZATION")
    print("=" * 80)
    
    from RL.community_detection import CommunityDetector
    
    store = EnhancedStore()
    detector = CommunityDetector(store)
    
    if not detector.load_cache():
        print("\n⚠ No community cache found!")
        print("Run: python -m RL.community_detection")
        return
    
    stats = detector.get_statistics()
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total communities: {stats['num_communities']}")
    print(f"  Total nodes: {stats['num_nodes']}")
    print(f"  Avg community size: {stats['avg_community_size']:.1f}")
    print(f"  Largest community: {stats['max_community_size']} nodes")
    print(f"  Smallest community: {stats['min_community_size']} nodes")
    
    print(f"\n Largest Communities:")
    
    sorted_comms = sorted(
        detector.community_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for i, (comm_id, size) in enumerate(sorted_comms, 1):
        percentage = (size / stats['num_nodes']) * 100
        print(f"  {i}. Community {comm_id}: {size} nodes ({percentage:.1f}%)")
    
    print(f"\n Community Connections (sample):")
    
    for comm_id, _ in sorted_comms[:3]:
        neighbors = await detector.get_community_neighbors(comm_id, limit=5)
        print(f"\n  Community {comm_id} connects to:")
        for neighbor_comm in neighbors:
            neighbor_size = detector.get_community_size(neighbor_comm)
            print(f"    → {neighbor_comm} ({neighbor_size} nodes)")
    
    await store.pool.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        asyncio.run(visualize_community_structure())
    else:
        asyncio.run(test_community_rewards())