import json 
import pickle 
from typing import Dict , List , Optional 
from collections import defaultdict
import numpy as np 
from datetime import datetime


class UserFeedbackTracker: 
    """Tracks and rewards based on historical user interactions."""

    def __init__(self , feedback_file: str = 'feedback.json'):
        self.feedback_file = feedback_file
        self.clicks = defaultdict(int)
        self.dwell_time = defaultdict(list)
        self.saves = defaultdict(int)
        self.citations = defaultdict(int)

        # Session tracking
        self.current_session = []
        self.session_history = []
        
        # Load existing feedback
        self.load_feedback()

    def load_feedback(self): 
        """Load historical feedback from disk."""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
                self.clicks = defaultdict(int, data.get('clicks', {}))
                self.dwell_time = defaultdict(list, data.get('dwell_time', {}))
                self.saves = defaultdict(int, data.get('saves', {}))
                self.citations = defaultdict(int, data.get('citations', {}))
                print(f"Loaded feedback for {len(self.clicks)} papers")
        except FileNotFoundError:
            print("No existing feedback file, starting fresh")
    
    def save_feedback(self):
        """Persist feedback to disk."""
        data = {
            'clicks': dict(self.clicks),
            'dwell_time': dict(self.dwell_time),
            'saves': dict(self.saves),
            'citations': dict(self.citations)
        }
        with open(self.feedback_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_clicks(self , paper_id: str ):
        self.clicks[paper_id] += 1 
        self.current_session.append({
            'paper_id' : paper_id,
            'action' : 'click',
            'timestamp' : datetime.now().isoformat()
        })

    def record_dwell(self , paper_id: str ,seconds: float): 
        self.dwell_time[paper_id].append(seconds)

        self.current_session.append({
            'paper_id' : paper_id,
            'action' : 'dwell',
            'duration' : seconds,
            'timestamp' : datetime.now().isoformat()
        })

    def record_save(self, paper_id: str):
        self.saves[paper_id] += 1
        self.current_session.append({
            'paper_id': paper_id,
            'action': 'save',
            'timestamp': datetime.now().isoformat()
        })
    
    def record_cite(self, paper_id: str):
        self.citations[paper_id] += 1
        self.current_session.append({
            'paper_id': paper_id,
            'action': 'cite',
            'timestamp': datetime.now().isoformat()
        })
         

    def get_feedback_reward(self , paper_id:str) -> float:
        reward =0.0 

        if paper_id in self.citations:
            reward += min(self.citations[paper_id] * 2.0 , 5.0 )
            
        if paper_id in self.saves: 
            reward += min(self.saves[paper_id] * 1.0 , 3.0 )
            
        if paper_id in self.dwell_time:
            avg_dwell = np.mean(self.dwell_time[paper_id])
            reward += min(avg_dwell / 30.0, 2.0) 
        
        if paper_id in self.clicks:
            reward += min(np.log10(self.clicks[paper_id] + 1), 1.0) 

        return reward
    
    def get_popularity_score(self , paper_id) -> float:     
        """Simple popularity metric for ranking."""
        clicks = self.clicks.get(paper_id, 0)
        saves = self.saves.get(paper_id, 0)
        cites = self.citations.get(paper_id, 0)
        
        return clicks + saves * 3 + cites * 10
        

    def end_session(self, query: str, retrieved_papers: List[str]):
        """Mark end of retrieval session."""
        self.session_history.append({
            'query': query,
            'retrieved': retrieved_papers,
            'interactions': self.current_session.copy(),
            'timestamp': datetime.now().isoformat()
        })
        self.current_session = []
        self.save_feedback()
        

    def simulate_feedback(self, paper_id: str, relevance_score: float):
        """
        Simulate user feedback for offline training.
        Based on relevance score (semantic similarity).
        """
        # Probabilistic simulation
        click_prob = min(relevance_score, 0.9)
        if np.random.random() < click_prob:
            self.record_clicks(paper_id)
            
            # If clicked, might also save
            save_prob = max(0, relevance_score - 0.5) * 2
            if np.random.random() < save_prob:
                self.record_save(paper_id)
                
                # If saved, might cite
                cite_prob = max(0, relevance_score - 0.7) * 3
                if np.random.random() < cite_prob:
                    self.record_cite(paper_id)
            
            dwell_seconds = relevance_score * 120  
            self.record_dwell(paper_id, dwell_seconds)