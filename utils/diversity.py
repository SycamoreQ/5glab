import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class DiversitySelector: 

    def __init__(self , lamdba_param: float = 0.7):
        self.lambda_param = lamdba_param


    def select_diverse_papers(self , candidates: List[Dict[str , Any]] ,query: np.ndarray, k: int = 10) -> List[Dict[str , Any]]:
        if len(candidates) <=k : 
            return candidates
        
        candidate_embeddings = np.array([c['embedding'] for c in candidates])
        relevance_score = cosine_similarity(candidate_embeddings , query.reshape(-1 , 1)).flatten()
        selected_indices = []
        selected_embeddings = []


        best_idx = np.argmax(relevance_score)
        selected_indices.append(best_idx)
        selected_embeddings.append(candidate_embeddings[best_idx])
        
        for _ in range(k-1): 
            remaining_idx = [i for i in range(len(selected_indices)) if i not in selected_indices]

            if not remaining_idx: 
                break 

            mmr_score = []
            for idx in remaining_idx: 
                relevance = relevance_score[idx]
                
                # Max similarity to already selected papers
                if selected_embeddings:
                    similarities = cosine_similarity(
                        [candidate_embeddings[idx]], 
                        selected_embeddings
                    ).flatten()
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_score.append(mmr)
            
            best_mmr_idx = remaining_idx[np.argmax(mmr_score)]
            selected_indices.append(best_mmr_idx)
            selected_embeddings.append(candidate_embeddings[best_mmr_idx])
        
        return [candidates[i] for i in selected_indices]


    def calculate_diversity_reward(
        self, 
        retrieved_papers: List[Dict[str, Any]]
    ) -> float:
        if len(retrieved_papers) < 2:
            return 0.0
        
        embeddings = np.array([p['embedding'] for p in retrieved_papers])
        sim_matrix = cosine_similarity(embeddings)
        
        n = len(sim_matrix)
        avg_similarity = (sim_matrix.sum() - n) / (n * (n - 1))
        
        diversity = 1.0 - avg_similarity
        
        return max(0, diversity) 

        
         
        

        
