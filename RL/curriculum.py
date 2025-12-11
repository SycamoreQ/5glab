import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class CurriculumManager:
    def __init__(self , cached_papers , encoder): 
        self.cached_papers = cached_papers
        self.encoder = encoder 
        self.current_stages = 0
        self.stage_episodes = [30 , 50 , 50 , 50 ]
        self.performance_history = deque(maxlen = 20)

        self.stages = [
            {
                'name': 'Stage 1: Direct Matches',
                'query_difficulty': 'easy',
                'max_steps': 10,
                'start_similarity_threshold': 0.7,  # Start from highly relevant papers
                'description': 'Find papers very similar to query'
            },
            {
                'name': 'Stage 2: Related Work',
                'query_difficulty': 'medium',
                'max_steps': 20,
                'start_similarity_threshold': 0.5,  # Moderate starting point
                'description': 'Navigate to related but not identical papers'
            },
            {
                'name': 'Stage 3: Cross-Domain',
                'query_difficulty': 'hard',
                'max_steps': 30,
                'start_similarity_threshold': 0.3,  # Challenging starts
                'description': 'Bridge different research areas'
            },
            {
                'name': 'Stage 4: Expert',
                'query_difficulty': 'expert',
                'max_steps': 50,
                'start_similarity_threshold': 0.2,  # Any starting point
                'description': 'Complex multi-hop reasoning'
            }
        ]
        
        self._prepare_curriculum_queries()


    def _prepare_curriculum_queries(self): 
        self.easy_queries = [
            "convolutional neural networks image classification",
            "BERT language model",
            "ResNet architecture",
            "Adam optimizer",
            "dropout regularization",
            "batch normalization",
            "attention mechanism transformers",
            "word embeddings word2vec",
            "recurrent neural networks LSTM",
            "gradient descent backpropagation"
        ]
        
        self.medium_queries = [
            "deep learning computer vision applications",
            "natural language processing sentiment analysis",
            "reinforcement learning robotics control",
            "graph neural networks molecular property prediction",
            "transfer learning few-shot learning",
            "generative models image synthesis",
            "neural architecture search optimization",
            "federated learning privacy preservation",
            "adversarial robustness defense mechanisms",
            "meta-learning learning to learn"
        ]
        
        self.hard_queries = [
            "applying computer vision techniques to medical diagnosis",
            "combining reinforcement learning with natural language",
            "graph methods for time series forecasting",
            "transfer learning from vision to language",
            "neural networks inspired by neuroscience",
            "explainability in deep learning systems",
            "quantum computing for machine learning acceleration",
            "edge AI and model compression",
            "causal inference in neural networks",
            "self-supervised learning across modalities"
        ]
        
        self.expert_queries = [
            "novel architectures for multimodal understanding",
            "theoretical foundations of deep learning generalization",
            "sample-efficient learning in complex environments",
            "bridging symbolic and neural approaches",
            "learning with limited supervision",
            "robustness and reliability in AI systems",
            "computational efficiency in large-scale models",
            "interpretability and transparency",
            "continual learning without catastrophic forgetting",
            "foundation models and emergent capabilities"
        ]


    def get_current_stage(self , episode:int) -> Dict: 
        cumulative_episodes = 0
        for i, stage_eps in enumerate(self.stage_episodes):
            cumulative_episodes += stage_eps
            if episode < cumulative_episodes:
                self.current_stage = i
                return self.stages[i]
        
        # Past all stages, stay at expert
        self.current_stage = len(self.stages) - 1
        return self.stages[-1]
    

    def get_query_for_stage(self, stage: Dict, episode: int) -> str:
        difficulty = stage['query_difficulty']
        
        if difficulty == 'easy':
            queries = self.easy_queries
        elif difficulty == 'medium':
            queries = self.medium_queries
        elif difficulty == 'hard':
            queries = self.hard_queries
        else:
            queries = self.expert_queries
        
        return queries[episode % len(queries)]
    

    def get_starting_paper(self , query: str , stage: Dict) -> Dict: 
        threshold = stage['start_similarity_threshold']
        query_emb = self.encoder.encode_with_cache(query ,cache_key=f"query_{query}")

        scored_papers= []
        for paper in self.cached_papers:
            paper_id = paper['paper_id']
            if paper_id in self.encoder.cache:
                    paper_emb = self.encoder.cache[paper_id]
                    sim = np.dot(query_emb, paper_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(paper_emb) + 1e-9
                    )
                    scored_papers.append((paper, sim))
            
            scored_papers.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and get random from candidates
            candidates = [p for p, s in scored_papers if s >= threshold]
            
            if not candidates:
                # Fall back to top papers if threshold too strict
                candidates = [p for p, s in scored_papers[:50]]
            
            return np.random.choice(candidates)
        

    def update_performance(self, episode_reward: float, max_similarity: float):
        self.performance_history.append({
            'reward': episode_reward,
            'similarity': max_similarity
        })
    
    def should_advance_stage(self) -> bool:
        if len(self.performance_history) < 15:
            return False
        
        # Check if agent is performing well
        recent_sims = [h['similarity'] for h in list(self.performance_history)[-15:]]
        avg_sim = np.mean(recent_sims)
        
        stage = self.stages[self.current_stage]
        
        # Thresholds for advancement
        if stage['query_difficulty'] == 'easy' and avg_sim > 0.6:
            return True
        elif stage['query_difficulty'] == 'medium' and avg_sim > 0.55:
            return True
        elif stage['query_difficulty'] == 'hard' and avg_sim > 0.5:
            return True
        
        return False
    
    def get_stage_summary(self) -> str:
        stage = self.stages[self.current_stage]
        if self.performance_history:
            recent_sims = [h['similarity'] for h in list(self.performance_history)[-10:]]
            avg_sim = np.mean(recent_sims)
            return (f"\n{'='*70}\n"
                    f"CURRICULUM: {stage['name']}\n"
                    f"Description: {stage['description']}\n"
                    f"Difficulty: {stage['query_difficulty'].upper()}\n"
                    f"Max Steps: {stage['max_steps']}\n"
                    f"Recent Performance: {avg_sim:.3f} avg similarity\n"
                    f"{'='*70}")
        return f"\nCURRICULUM: {stage['name']}"