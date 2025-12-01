"""
Quick test to verify a single episode runs successfully with non-zero rewards.
"""
import asyncio
from graph.database.store import EnhancedStore
from RL.env import AdvancedGraphTraversalEnv, RelationType
from RL.ddqn import DDQLAgent

async def test_single_episode():
    print("=" * 80)
    print("TESTING SINGLE EPISODE")
    print("=" * 80)
    
    # Initialize
    store = EnhancedStore()
    env = AdvancedGraphTraversalEnv(store)
    agent = DDQLAgent(773, 384)
    
    # Get a starting paper with good connectivity
    print("\n1. Fetching well-connected paper...")
    paper = await store.get_well_connected_paper()
    
    if not paper:
        print("✗ No well-connected papers in database!")
        print("  Trying any paper...")
        paper = await store.get_any_paper()
        
        if not paper:
            print("✗ No papers in database at all!")
            return
    
    paper_id = paper.get('paper_id')
    title = paper.get('title', '')
    original_id = paper.get('original_id', '')
    ref_count = paper.get('ref_count', 0)
    cite_count = paper.get('cite_count', 0)
    
    # Use actual text for display and query
    display_text = title or original_id or paper_id or 'Unknown'
    
    print(f"✓ Starting paper: {display_text[:60]}...")
    print(f"  ID: {paper_id}")
    print(f"  Title field: '{title}'")
    print(f"  Original_id field: '{original_id}'")
    print(f"  References: {ref_count}, Citations: {cite_count}")
    
    # Reset environment with a meaningful query
    print("\n2. Resetting environment...")
    initial_intent = RelationType.CITED_BY
    
    # Use the paper's title/id as the query for semantic matching
    query_text = title or original_id or 'machine learning research'
    print(f"  Query text: '{query_text[:60]}...'")
    
    state = await env.reset(query_text, initial_intent, start_node_id=paper_id)
    print(f"✓ State shape: {state.shape}")
    print(f"  Expected: (773,)")
    
    # Run a few steps
    print("\n3. Running episode...")
    total_reward = 0.0
    step = 0
    done = False
    
    while not done and step < 3:
        step += 1
        print(f"\n--- Step {step} ---")
        
        # Manager: Get valid actions
        manager_actions = await env.get_manager_actions()
        print(f"Manager actions available: {len(manager_actions)}")
        
        if not manager_actions:
            print("✗ No manager actions!")
            break
        
        # Remove STOP for testing
        if RelationType.STOP in manager_actions and len(manager_actions) > 1:
            manager_actions.remove(RelationType.STOP)
        
        # Choose random manager action
        import random
        manager_action = random.choice(manager_actions)
        print(f"Manager chose: {manager_action}")
        
        # Execute manager step
        is_terminal, manager_reward = await env.manager_step(manager_action)
        print(f"Manager reward: {manager_reward:.4f}")
        total_reward += manager_reward
        
        if is_terminal:
            print("Episode terminated by manager")
            break
        
        # Worker: Get valid actions
        worker_actions = await env.get_worker_actions()
        print(f"Worker actions available: {len(worker_actions)}")
        
        if not worker_actions:
            print("✗ No worker actions!")
            continue

        chosen_node, _ = random.choice(worker_actions)
        node_text = (
            chosen_node.get('title') or 
            chosen_node.get('original_id') or 
            chosen_node.get('paper_id') or
            'Unknown'
        )
        print(f"Worker chose: {node_text[:50]}...")
        
        if node_text == 'Unknown' or not chosen_node.get('title'):
            print(f"  DEBUG - Node fields: {list(chosen_node.keys())}")
            print(f"  DEBUG - Sample values: {dict(list(chosen_node.items())[:3])}")
        
        next_state, worker_reward, done = await env.worker_step(chosen_node)
        print(f"Worker reward: {worker_reward:.4f}")
        print(f"Total step reward: {manager_reward + worker_reward:.4f}")
        
        total_reward += worker_reward
        state = next_state
    
    print("\n" + "=" * 80)
    print("EPISODE COMPLETE")
    print("=" * 80)
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    
    if total_reward != 0:
        print("\n✓✓✓ SUCCESS! Non-zero rewards achieved! ✓✓✓")
    else:
        print("\n✗✗✗ FAILED: Still getting zero rewards ✗✗✗")
    
    await store.pool.close()

if __name__ == "__main__":
    asyncio.run(test_single_episode())