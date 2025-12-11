import torch
import torch.nn as nn
import numpy as np


class CuriosityModule:
    def __init__(self, state_dim=773, embedding_dim=384):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=1e-4)
        self.visited_papers = set()
        
    def compute_intrinsic_reward(self, state, action_emb, next_state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action_emb).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        predicted_next_state = self.forward_model(torch.cat([state_t, action_t], dim=1))
        prediction_error = torch.mean((predicted_next_state - next_state_t) ** 2)
        intrinsic_reward = prediction_error.item()
        
        loss = prediction_error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return min(intrinsic_reward * 10, 20.0)
    
    def get_novelty_bonus(self, paper_id):
        if paper_id not in self.visited_papers:
            self.visited_papers.add(paper_id)
            return 5.0
        return 0.0
