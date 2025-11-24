import ray 
import asyncio 
from env import AdvancedGraphTraversalEnv , RelationType
from ddqn import DDQLAgent 
import random

if __name__ == "__main__":
    # Connect to the existing Ray cluster
    ray.init(address="auto") 
    print("Connected to Ray Cluster")

    # 1. Start Storage (Likely on Head Node B)
    storage = SharedStorage.remote()

    # 2. Start Learner (Forces it to run where GPU is available, Node B)
    learner = Learner.remote(storage)
    learner_task = learner.update_model.remote()

    # 3. Start Workers (Spread across cluster, ideally Node A)
    # We can use 'resources' to force placement if we tagged Node A with custom resources
    workers = [GraphExplorer.remote(i) for i in range(NUM_WORKERS)]
    worker_tasks = [w.run_exploration.remote(storage) for w in workers]

    # Keep running
    ray.get(learner_task)
    