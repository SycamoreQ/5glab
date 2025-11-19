import torch
import torch.multiprocessing as mp
import asyncio
import logging
import random
import time
import os
from collections import deque

import DDQLAgent, env, RelationType
from graph.database.store import EnhancedStore

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(processName)s - %(message)s'
)

NUM_WORKERS = 4          # Number of parallel graph explorers
UPDATE_EVERY = 50        # How often workers copy weights from learner
SYNC_TARGET_EVERY = 100  # How often learner updates target net
MAX_TIME_SECONDS = 2000   # Train for 10 minutes (example)

def worker_process(worker_id, shared_queue, shared_model_dict, stop_event):
    """
    Worker: Explores the graph using a local copy of the model.
    Pushes transitions (State, Action, Reward...) to the Shared Queue.
    """
    # IMPORTANT: Re-initialize store inside process (DB connections aren't picklable)
    # Use a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_worker():
        logging.info(f"Worker {worker_id} started.")
        store = EnhancedStore() # New connection
        env = env(store)
        
        # Local Agent Copy
        # We only need the policy net for inference
        agent = DDQLAgent(state_dim=773, text_dim=384)
        agent.epsilon = 0.4 + (worker_id * 0.1)  # Diversity: Workers have different exploration rates
        
        # Load initial weights
        if shared_model_dict:
            agent.policy_net.load_state_dict(shared_model_dict)
        
        steps = 0
        
        # Training Scenarios
        training_scenarios = [
            ("deep learning transformers", RelationType.CITES),
            ("graph neural networks", RelationType.CITED_BY),
            ("yann lecun", RelationType.WROTE),
            ("reinforcement learning", RelationType.CITES)
        ]

        while not stop_event.is_set():
            # 1. Sync weights periodically from Learner
            if steps % UPDATE_EVERY == 0 and len(shared_model_dict) > 0:
                 try:
                    # Load state from shared dict (CPU-based sync)
                    # We convert the DictProxy back to a real dict
                    state_dict = {k: v for k, v in shared_model_dict.items()}
                    agent.policy_net.load_state_dict(state_dict)
                 except Exception as e:
                     logging.debug(f"Worker sync skip: {e}")

            # 2. Start Episode
            query, intent = random.choice(training_scenarios)
            try:
                state = await env.reset(query, intent)
            except Exception:
                continue # Retry if random query fails
                
            done = False
            while not done:
                # 3. Act
                actions = await env.get_valid_actions()
                if not actions: break
                
                action_tuple = agent.act(state, actions) 
                if not action_tuple: break
                
                node, relation = action_tuple
                next_state, reward, done = await env.step(node, relation)
                next_actions = await env.get_valid_actions()
                
                # 4. Push Experience to Central Learner
                # We strip heavy objects if needed, but here we send the tuple
                # Queue is thread-safe
                try:
                    experience = (state, action_tuple, reward, next_state, done, next_actions)
                    shared_queue.put(experience)
                except Exception as e:
                    logging.error(f"Worker {worker_id} queue full/error: {e}")
                    
                state = next_state
                steps += 1
                
                if stop_event.is_set(): break
    
    # Run the async loop
    loop.run_until_complete(run_worker())


def learner_process(shared_queue, shared_model_dict, stop_event):
    """
    Learner: Pops experiences from Queue, trains the network, updates shared weights.
    """
    logging.info("Learner Process started.")
    
    # The Master Agent
    agent = DDQLAgent(state_dim=773, text_dim=384)
    agent.epsilon = 0.1 # Low epsilon, just for consistency
    
    steps = 0
    
    while not stop_event.is_set():
        try:
            # 1. Fetch Experience
            # get() blocks until item is available
            experience = shared_queue.get(timeout=1)
            
            state, action, reward, next_state, done, next_actions = experience
            
            # 2. Store in Replay Buffer
            agent.remember(state, action, reward, next_state, done, next_actions)
            
            # 3. Train
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                steps += 1
                
                # 4. Sync Weights back to Workers
                if steps % UPDATE_EVERY == 0:
                    # Update the shared dict so workers can pull it
                    # Must move to CPU for multiprocessing compatibility
                    cpu_state = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
                    shared_model_dict.update(cpu_state)
                
                # 5. Update Target Network
                if steps % SYNC_TARGET_EVERY == 0:
                    agent.update_target()
                    logging.info(f"Learner Step {steps}: Target Net Updated")

        except Exception:
            # Queue empty or timeout, just loop
            continue

    # Save final model
    torch.save(agent.policy_net.state_dict(), "distributed_ddql_navigator.pth")
    logging.info("Training finished. Model saved.")


if __name__ == "__main__":
    # Required for PyTorch Multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # Shared Data Structures
    manager = mp.Manager()
    
    # Queue: Workers put data here, Learner gets data here
    shared_queue = manager.Queue(maxsize=10000)
    
    # Dictionary: Learner updates weights here, Workers read from here
    shared_model_dict = manager.dict()
    
    stop_event = mp.Event()
    
    # Initialize shared dict with random weights from a fresh agent
    init_agent = DDQLAgent(state_dim=773, text_dim=384)
    for k, v in init_agent.policy_net.state_dict().items():
        shared_model_dict[k] = v

    processes = []

    # Start Learner (1 Process)
    p_learner = mp.Process(
        target=learner_process, 
        args=(shared_queue, shared_model_dict, stop_event)
    )
    p_learner.start()
    processes.append(p_learner)

    # Start Workers (N Processes)
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=worker_process, 
            args=(i, shared_queue, shared_model_dict, stop_event)
        )
        p.start()
        processes.append(p)

    try:
        print(f"Training for {MAX_TIME_SECONDS} seconds...")
        time.sleep(MAX_TIME_SECONDS)
    except KeyboardInterrupt:
        logging.info("Stopping manually...")
    
    stop_event.set()
    for p in processes:
        p.join()
        
    print("All processes stopped.")