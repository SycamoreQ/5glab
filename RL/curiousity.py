import torch
import torch.nn as nn
import numpy as np

class CuriosityModule:
    def __init__(self, state_dim=783, embedding_dim=384): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        input_dim = state_dim + embedding_dim  
        self.forward_model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, state_dim)  # Output matches state_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=1e-4)
        self.visited_papers = set()
        self.prediction_errors = [] 
        
    def compute_intrinsic_reward(self, state, action_emb, next_state):
        """
        Compute intrinsic reward based on prediction error.
        High error = novel/surprising transition = high reward.
        """
        # Validate input dimensions
        if not isinstance(state, np.ndarray) or len(state) != self.state_dim:
            print(f"[WARN] Invalid state dim: {len(state) if isinstance(state, np.ndarray) else type(state)}, expected {self.state_dim}")
            return 0.0
            
        if not isinstance(action_emb, np.ndarray) or len(action_emb) != self.embedding_dim:
            print(f"[WARN] Invalid action_emb dim: {len(action_emb) if isinstance(action_emb, np.ndarray) else type(action_emb)}, expected {self.embedding_dim}")
            return 0.0
        
        if not isinstance(next_state, np.ndarray) or len(next_state) != self.state_dim:
            print(f"[WARN] Invalid next_state dim: {len(next_state) if isinstance(next_state, np.ndarray) else type(next_state)}, expected {self.state_dim}")
            return 0.0
        
        try:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action_emb).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Predict next state
            predicted_next_state = self.forward_model(torch.cat([state_t, action_t], dim=1))
            
            # Compute prediction error (MSE)
            prediction_error = torch.mean((predicted_next_state - next_state_t) ** 2)
            intrinsic_reward = prediction_error.item()
            
            # Train forward model
            loss = prediction_error
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0) 
            self.optimizer.step()
            
            # Track prediction errors for monitoring
            self.prediction_errors.append(intrinsic_reward)
            if len(self.prediction_errors) > 100:
                self.prediction_errors.pop(0)
            
            # Scale and cap intrinsic reward
            scaled_reward = min(intrinsic_reward * 10, 20.0)
            return scaled_reward
            
        except Exception as e:
            print(f"[ERROR] CuriosityModule exception: {e}")
            return 0.0
    
    def get_novelty_bonus(self, paper_id):
        """Simple novelty bonus for first-time visits."""
        if paper_id not in self.visited_papers:
            self.visited_papers.add(paper_id)
            return 5.0
        return 0.0
    
    def get_avg_prediction_error(self):
        """Get average prediction error for monitoring."""
        if not self.prediction_errors:
            return 0.0
        return np.mean(self.prediction_errors)
    
    def reset(self):
        """Reset visited papers for new episode."""
        self.visited_papers.clear()
