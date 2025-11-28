import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
import torch.functional as F


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
        self.device = torch.device
        
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

    def predict_q_value(self , state , action_emb , relation_onehot):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_emb_t = torch.FloatTensor(action_emb).unsqueeze(0).to(self.device)
        relation_onehot_t = torch.FloatTensor(relation_onehot).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad:
            q_value = self.policy_net(state_t , action_emb_t , relation_onehot_t)
        
        self.policy_net.train
        return q_value      
    

    def replay_batch(self, batch):
        """
        Performs a single optimization step using a batch from SharedStorage.
        The batch is expected to have the structure:
        (state, action_emb, relation_type, reward, next_state, done, next_actions_list)
        """
        if not batch:
            return 0.0 

        states = []
        action_embs = []
        relation_onehots_np = []
        rewards = []
        next_states = []
        dones = []
        target_action_embs = []
        target_relation_onehots_np = []
        target_q_values = []
    
        self.policy_net.eval()
        self.target_net.eval()

        for state, act_emb, r_type, reward, next_state, done, next_actions in batch:
            states.append(state)
            action_embs.append(act_emb)
            relation_onehots_np.append(self._get_relation_onehot(r_type).squeeze(0).cpu().numpy())
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            if next_actions and not done:
                best_next_q_policy = -float('inf')
                best_next_emb = np.zeros(self.text_dim)
                best_next_rel = 0 
                
                for n_node, n_r_type in next_actions:
                    n_txt = n_node.get('title', '') or n_node.get('name', '')
                    n_emb = self.encoder.encode(n_txt)
                    
                    # Prepare inputs
                    next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    n_emb_t = torch.FloatTensor(n_emb).unsqueeze(0).to(self.device)
                    n_rel_onehot_t = self._get_relation_onehot(n_r_type)
                    
                    with torch.no_grad():
                        q_value_policy = self.policy_net(next_state_t, n_emb_t, n_rel_onehot_t).item()
                        
                        if q_value_policy > best_next_q_policy:
                            best_next_q_policy = q_value_policy
                            best_next_emb = n_emb
                            best_next_rel = n_r_type
                
                target_action_embs.append(best_next_emb)
                target_relation_onehots_np.append(self._get_relation_onehot(best_next_rel).squeeze(0).cpu().numpy())

            else:
                target_action_embs.append(np.zeros(self.text_dim))
                target_relation_onehots_np.append(np.zeros(self.relation_dim))
        
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs_t = torch.FloatTensor(np.array(action_embs)).to(self.device)
        relation_onehots_t = torch.FloatTensor(np.array(relation_onehots_np)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        target_action_embs_t = torch.FloatTensor(np.array(target_action_embs)).to(self.device)
        target_relation_onehots_t = torch.FloatTensor(np.array(target_relation_onehots_np)).to(self.device)


        self.policy_net.train()
        self.target_net.train()

        q_values = self.policy_net(states_t, action_embs_t, relation_onehots_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t, target_action_embs_t, target_relation_onehots_t)

            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
        
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        return loss.item()
        
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())