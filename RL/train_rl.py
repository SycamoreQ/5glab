import asyncio
import logging
import random
import torch
import numpy as np 
import env, DDQLAgent , RelationType
from graph.database.store import EnhancedStore

logging.basicConfig(level=logging.INFO)

async def train_advanced():
    store = EnhancedStore()
    env = env(store)
    agent = DDQLAgent(state_dim=773, text_dim=384) 
    episodes = 200
    training_scenarios = [
        ("deep learning transformers", RelationType.CITES),    # Intent: Find references
        ("graph neural networks", RelationType.CITED_BY),      # Intent: Find papers citing this
        ("yann lecun", RelationType.WROTE),                    # Intent: Find author's works
        ("reinforcement learning", RelationType.CITES),
        ("attention mechanism", RelationType.CITED_BY),
        ("geoffrey hinton", RelationType.WROTE)
    ]

    print("Starting Advanced Intent-Aware RL Training...")

    for e in range(episodes):
        query_text, intent_type = random.choice(training_scenarios)
        intent_name = RelationType(intent_type).name
        
        try:
            state = await env.reset(query_text, intent_type)
        except ValueError as err:
            print(f"Skipping episode {e}: {err}")
            continue
            
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            valid_actions = await env.get_valid_actions()
            
            if not valid_actions:
                break
            
            action_tuple = agent.act(state, valid_actions)
            
            if not action_tuple:
                break

            action_node, action_relation = action_tuple
            
            next_state, reward, done = await env.step(action_node, action_relation)
            
            next_actions = await env.get_valid_actions()
            
            agent.remember(state, action_tuple, reward, next_state, done, next_actions)
            
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        if e % 10 == 0:
            agent.update_target()
            
        print(f"Episode {e+1}/{episodes} | Query: '{query_text}' | Intent: {intent_name} | Steps: {steps} | Reward: {total_reward:.4f} | Eps: {agent.epsilon:.2f}")

    torch.save(agent.policy_net.state_dict(), "ddql_navigator.pth")
    print("Training complete. Advanced model saved.")

if __name__ == "__main__":
    asyncio.run(train_advanced())