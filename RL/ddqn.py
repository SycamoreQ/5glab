import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_emb_dim, relation_dim=5):
        super(QNetwork, self).__init__()
        
        # Input: State + ActionNode_Emb + Relation_OneHot
        self.input_dim = state_dim + action_emb_dim + relation_dim
        
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action_emb, relation_onehot):
        x = torch.cat([state, action_emb, relation_onehot], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    


class DDQLAgent: 
    def __init__(self, state_dim, text_dim):
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.relation_dim = 5
        
        self.policy_net = QNetwork(state_dim, text_dim, self.relation_dim)
        self.target_net = QNetwork(state_dim, text_dim, self.relation_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=5000)
        
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_relation_onehot(self, r_type):
        t = torch.zeros(1, self.relation_dim)
        t[0][r_type] = 1.0
        return t
    
    def act(self , state , valid_actions):
        if not valid_actions: 
            return None 
        
        if np.random.rand < self.epsilon: 
            return random.choise(valid_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        best_q = -float('inf')
        best_action = None # (node, type)
        

        with torch.no_grad():
            for node , r_type in valid_actions:
                node_text = f"{node.get('title', '')} {node.get('abstract', '')}"
                node_emb = self.encoder.encode(node_text)
                node_emb_t = torch.FloatTensor(node_emb).unsqueeze(0)
                relation_onehot_t = self._get_relation_onehot(r_type)
                
                q_value = self.policy_net(state_t, node_emb_t, relation_onehot_t)
                if q_value.item() > best_q:
                    best_q = q_value.item()
                    best_action = (node, r_type)

        return best_action
    

    def remember(self, state, action_tuple, reward, next_state, done, next_actions):
        node, r_type = action_tuple
        txt = node.get('title') or node.get('name', '')
        act_emb = self.encoder.encode(txt)
        
        # Store simplified experience to save RAM
        self.memory.append((state, act_emb, r_type, reward, next_state, done, next_actions))


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        action_embs = []
        relation_onehots = []
        rewards = []
        next_states = []
        dones = []
        next_action_embs = []
        next_relation_onehots = []
        
        for state, act_emb, r_type, reward, next_state, done, next_actions in batch:
            states.append(state)
            action_embs.append(act_emb)
            relation_onehots.append(self._get_relation_onehot(r_type).squeeze(0).numpy())
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # For next actions,just pick the best Q-value among them
            if next_actions:
                best_next_q = -float('inf')
                best_next_emb = None
                best_next_rel = None
                for n_node, n_r_type in next_actions:
                    n_txt = n_node.get('title', '') or n_node.get('name', '')
                    n_emb = self.encoder.encode(n_txt)
                    n_emb_t = torch.FloatTensor(n_emb).unsqueeze(0)
                    n_rel_onehot_t = self._get_relation_onehot(n_r_type)
                    
                    with torch.no_grad():
                        q_value = self.target_net(torch.FloatTensor(next_state).unsqueeze(0), n_emb_t, n_rel_onehot_t)
                        if q_value.item() > best_next_q:
                            best_next_q = q_value.item()
                            best_next_emb = n_emb
                            best_next_rel = n_r_type
                next_action_embs.append(best_next_emb)
                next_relation_onehots.append(self._get_relation_onehot(best_next_rel).squeeze(0).numpy())
            else:
                next_action_embs.append(np.zeros(self.text_dim))
                next_relation_onehots.append(np.zeros(self.relation_dim))
    
        states_t = torch.FloatTensor(states)
        action_embs_t = torch.FloatTensor(action_embs)
        relation_onehots_t = torch.FloatTensor(relation_onehots)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        next_action_embs_t = torch.FloatTensor(next_action_embs)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())