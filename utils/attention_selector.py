import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class GraphAttentionSelector(nn.Module): 
    def __init__(self , embed_dim: int = 384 , hidden_dim:int = 256 , num_heads:int = 4): 
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        
        self.relation_embed = nn.Embedding(13, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self , query_emb , candidate_emb , relation_type = None , current_state = None):
        batch_size = 1 
        num_candidates = len(candidate_emb)
        
        if num_candidates == 0:
            return torch.Tensor([])
        
        query_tensor = torch.FloatTensor(query_emb).unsqueeze(0)
        candidate_tensor = torch.FloatTensor(candidate_emb).unsqueeze(0)
        
        Q = self.query_proj(query_tensor)
        K = self.key_proj(candidate_tensor)
        V = self.value_proj(candidate_tensor)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(num_candidates, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        V = V.view(num_candidates, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        scores = torch.matmul(Q , K.transpose(-2 , -1))/np.sqrt(self.head_dim) + 1e-6

        if relation_type is not None : 
            rel_tensor = torch.LongTensor(relation_type)
            rel_emb = self.relation_embed(rel_tensor)
            rel_emb = rel_emb.view(num_candidates, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            rel_scores = torch.matmul(Q, rel_emb.transpose(-2, -1)) / np.sqrt(self.head_dim)
            scores = scores + 0.3 * rel_scores
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        
        if current_state is not None:
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).unsqueeze(1)
            state_proj = self.query_proj(state_tensor[:, :, :self.embed_dim])
            attn_output = self.layer_norm(attn_output + state_proj)
        
        scores = self.out_proj(attn_output).squeeze()
        
        return torch.softmax(scores, dim=0).detach().numpy()
    

class CommunityAwareAttention(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.candidate_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.community_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def encode_community_context(self , community_id , community_history , community_sizes): 
        if community_id is None:
            return torch.Tensor(256)
        
        visit_count = community_history.count(community_id)
        size = community_sizes.get(community_id , 100)
        
        recency = 0.0 
        if community_id in community_history:
            last_idx = len(community_history) - 1 - community_history[::-1].index(community_id)
            recency = 1.0 / (len(community_history) - last_idx + 1)

            features = torch.FloatTensor([
            visit_count / 10.0,
            np.log10(size + 1) / 3.0,
            recency,
            1.0 if visit_count > 0 else 0.0,
            min(size / 200.0, 1.0)
        ])
            

        expanded = features.unsqueeze(0).repeat(1, 256 // 5)
        return expanded.view(-1)[:256]
    
    def forward(self, query_emb, candidate_embs, candidate_communities, 
            current_community, community_history, community_sizes):
    
        num_candidates = len(candidate_embs)
        if num_candidates == 0:
            return torch.tensor([])
        
        query_tensor = torch.FloatTensor(query_emb)
        candidate_tensor = torch.FloatTensor(np.array(candidate_embs))
        
        query_repr = self.query_net(query_tensor).unsqueeze(0).expand(num_candidates, -1)
        candidate_repr = self.candidate_net(candidate_tensor)
        
        community_reprs = []
        for comm_id in candidate_communities:
            comm_ctx = self.encode_community_context(comm_id, community_history, community_sizes)
            community_reprs.append(comm_ctx)
        community_repr = torch.stack(community_reprs)
        community_repr = self.community_net(community_repr)
        
        combined = torch.cat([query_repr, candidate_repr, community_repr], dim=1)
        scores = self.score_net(combined).squeeze()
        
        if current_community is not None:
            for i, comm_id in enumerate(candidate_communities):
                if comm_id == current_community:
                    scores[i] = scores[i] - 0.5
                elif comm_id not in community_history:
                    scores[i] = scores[i] + 0.8
        
        return torch.softmax(scores, dim=0).detach().numpy()
        
    
class HybridAttentionSelector:
    def __init__(self, embed_dim=384, use_community_attention=True):
        self.embed_dim = embed_dim
        self.use_community_attention = use_community_attention
        
        self.graph_attention = GraphAttentionSelector(embed_dim)
        
        if use_community_attention:
            self.community_attention = CommunityAwareAttention(embed_dim)
        
        self.mode = 'hybrid'
        
    def select_action(self, query_emb, candidate_nodes, candidate_embs, relation_types,
                      current_state=None, current_community=None, community_history=None,
                      community_sizes=None, community_detector=None):
        
        if len(candidate_nodes) == 0:
            return None, None, 0.0
        
        graph_scores = self.graph_attention(
            query_emb, 
            candidate_embs, 
            relation_types,
            current_state
        )
        
        if self.use_community_attention and community_detector is not None:
            candidate_communities = []
            for node in candidate_nodes:
                node_id = node.get('paper_id') or node.get('author_id')
                comm = community_detector.get_community(node_id) if node_id else None
                candidate_communities.append(comm)
            
            if community_history is not None and community_sizes is not None:
                community_scores = self.community_attention(
                    query_emb,
                    candidate_embs,
                    candidate_communities,
                    current_community,
                    community_history,
                    community_sizes
                )
                
                final_scores = 0.6 * graph_scores + 0.4 * community_scores
            else:
                final_scores = graph_scores
        else:
            final_scores = graph_scores
        
        if len(final_scores) == 0:
            return None, None, 0.0
        
        best_idx = np.argmax(final_scores)
        best_node = candidate_nodes[best_idx]
        best_relation = relation_types[best_idx] if relation_types else None
        best_score = float(final_scores[best_idx])
        
        return best_node, best_relation, best_score
    
    def get_ranked_actions(self, query_emb, candidate_nodes, candidate_embs, relation_types,
                           current_state=None, current_community=None, community_history=None,
                           community_sizes=None, community_detector=None, top_k=5):
        
        if len(candidate_nodes) == 0:
            return []
        
        graph_scores = self.graph_attention(
            query_emb, 
            candidate_embs, 
            relation_types,
            current_state
        )
        
        if self.use_community_attention and community_detector is not None:
            candidate_communities = []
            for node in candidate_nodes:
                node_id = node.get('paper_id') or node.get('author_id')
                comm = community_detector.get_community(node_id) if node_id else None
                candidate_communities.append(comm)
            
            if community_history is not None and community_sizes is not None:
                community_scores = self.community_attention(
                    query_emb,
                    candidate_embs,
                    candidate_communities,
                    current_community,
                    community_history,
                    community_sizes
                )
                
                final_scores = 0.6 * graph_scores + 0.4 * community_scores
            else:
                final_scores = graph_scores
        else:
            final_scores = graph_scores
        
        if len(final_scores) == 0:
            return []
        
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        ranked_actions = []
        for idx in top_indices:
            ranked_actions.append({
                'node': candidate_nodes[idx],
                'relation': relation_types[idx] if relation_types else None,
                'score': float(final_scores[idx]),
                'rank': len(ranked_actions) + 1
            })
        
        return ranked_actions