import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .prioritized_replay import PrioritizedReplay


class QNetwork(nn.Module):
    """Q-Network with dropout for regularization."""
    
    def __init__(self, state_dim, action_emb_dim, relation_dim=13):
        super(QNetwork, self).__init__()
        
        # Input: State + ActionNode_Emb + Relation_OneHot
        self.input_dim = state_dim + action_emb_dim + relation_dim
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action_emb, relation_onehot):
        """Forward pass through network."""
        x = torch.cat([state, action_emb, relation_onehot], dim=1)
        return self.network(x)


class DDQLAgent:
    """Double DQN Agent with Prioritized Experience Replay."""
    
    def __init__(self, state_dim=773, text_dim=384, use_prioritized=True):
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.relation_dim = 13
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_dim, text_dim, self.relation_dim).to(self.device)
        self.target_net = QNetwork(state_dim, text_dim, self.relation_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.5
        )
        
        # Memory
        self.use_prioritized = use_prioritized
        if use_prioritized:
            self.memory = PrioritizedReplay(
                capacity=50000,
                alpha=0.6,
                beta=0.4,
            )
        else:
            self.memory = deque(maxlen=50000)
        
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.2  
        self.epsilon_decay = 0.998 
        
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"DDQN Agent initialized on {self.device}")
        print(f"Prioritized Replay: {use_prioritized}")
    
    def _get_relation_onehot(self, r_type: int) -> torch.Tensor:
        """Convert relation type to one-hot encoding."""
        onehot = torch.zeros(1, self.relation_dim)
        if 0 <= r_type < self.relation_dim:
            onehot[0][r_type] = 1.0
        else:
            print(f"⚠ Invalid relation type {r_type}, using 0")
            onehot[0][0] = 1.0
        return onehot
    
    def act(self, state: np.ndarray, valid_actions: List[Tuple[Dict, int]]) -> Optional[Tuple[Dict, int]]:
        """Select action using epsilon-greedy policy."""
        if not valid_actions:
            return None
        
        memory_size = len(self.memory) if self.use_prioritized else len(self.memory)
        
        if len(self.memory) < 640:  
            return random.choice(valid_actions)
        
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action selection
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        best_q = -float('inf')
        best_action = None
        
        self.policy_net.eval()
        with torch.no_grad():
            for node, r_type in valid_actions:
                node_text = f"{node.get('title', '')} {node.get('abstract', '')}"
                node_emb = self.encoder.encode(node_text, convert_to_numpy=True)
                node_emb_t = torch.FloatTensor(node_emb).unsqueeze(0).to(self.device)
            
                relation_onehot_t = self._get_relation_onehot(r_type).to(self.device)
                
                q_value = self.policy_net(state_t, node_emb_t, relation_onehot_t)
                
                if q_value.item() > best_q:
                    best_q = q_value.item()
                    best_action = (node, r_type)
        
        self.policy_net.train()
        return best_action
    
    def remember(self, state, action_tuple, reward, next_state, done, next_actions):
        """Store experience in replay buffer."""
        node, r_type = action_tuple
        
        # Encode action node
        txt = node.get('title', '') or node.get('name', '')
        act_emb = self.encoder.encode(txt, convert_to_numpy=True)
        
        if self.use_prioritized:
            self.memory.add(state, act_emb,r_type, reward, next_state, done, next_actions)
        else:
            self.memory.append((state, act_emb, r_type  , reward, next_state, done, next_actions))
    
    def replay_prioritized(self) -> float:
        """Train with prioritized experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        if not batch:
            return 0.0
        
        states = []
        action_embs = []
        relation_onehots = []
        rewards = []
        next_states = []
        dones = []
        next_action_embs = []
        next_relation_onehots = []
        
        self.policy_net.eval()
        self.target_net.eval()
        
        for experience in batch:
            state, act_emb, r_type , reward , next_state, done, next_actions = experience
            
            states.append(state)
            action_embs.append(act_emb)
            relation_onehots.append(self._get_relation_onehot(r_type).squeeze(0).numpy())
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            if next_actions and not done:
                best_next_q = -float('inf')
                best_next_emb = np.zeros(self.text_dim)
                best_next_rel = 0
                
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    for n_node, n_r_type in next_actions:
                        n_txt = n_node.get('title', '') or n_node.get('name', '')
                        n_emb = self.encoder.encode(n_txt, convert_to_numpy=True)
                        n_emb_t = torch.FloatTensor(n_emb).unsqueeze(0).to(self.device)
                        n_rel_onehot_t = self._get_relation_onehot(n_r_type).to(self.device)
                        
                        q_policy = self.policy_net(next_state_t, n_emb_t, n_rel_onehot_t).item()
                        
                        if q_policy > best_next_q:
                            best_next_q = q_policy
                            best_next_emb = n_emb
                            best_next_rel = n_r_type
                
                next_action_embs.append(best_next_emb)
                next_relation_onehots.append(self._get_relation_onehot(best_next_rel).squeeze(0).numpy())
            else:
                next_action_embs.append(np.zeros(self.text_dim))
                next_relation_onehots.append(np.zeros(self.relation_dim))
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs_t = torch.FloatTensor(np.array(action_embs)).to(self.device)
        relation_onehots_t = torch.FloatTensor(np.array(relation_onehots)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        next_action_embs_t = torch.FloatTensor(np.array(next_action_embs)).to(self.device)
        next_relation_onehots_t = torch.FloatTensor(np.array(next_relation_onehots)).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        self.policy_net.train()
        q_values = self.policy_net(states_t, action_embs_t, relation_onehots_t)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t, next_action_embs_t, next_relation_onehots_t)
            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
        
        td_errors = (q_values - target_q_values).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)
        
        loss = (weights_t * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
    
        self.optimizer.zero_grad()
        loss.backward()
    
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def replay(self) -> float:
        if not self.use_prioritized:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            batch = random.sample(list(self.memory), self.batch_size)
            return self.replay_prioritized()
        else:
            return self.replay_prioritized()
    
    def update_target(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("  [Target network updated]")
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }, filepath)
        print(f" Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        print(f"✓ Model loaded from {filepath}")
        print(f"  Epsilon: {self.epsilon:.4f}")
