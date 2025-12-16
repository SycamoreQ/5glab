import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class GraphAttentionSelector(nn.Module): 
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 256, num_heads: int = 4): 
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

    def forward(self, query_emb, candidate_emb, relation_type=None, current_state=None):
        batch_size = 1
        num_candidates = len(candidate_emb)
        
        if num_candidates == 0:
            return torch.tensor([])
        
        # Validate embeddings
        normalized_candidates = []
        for i, emb in enumerate(candidate_emb):
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            emb = np.array(emb).flatten()
            if emb.shape[0] != self.embed_dim:
                if emb.shape[0] < self.embed_dim:
                    emb = np.pad(emb, (0, self.embed_dim - emb.shape[0]))
                else:
                    emb = emb[:self.embed_dim]
            normalized_candidates.append(emb)
        
        candidate_array = np.stack(normalized_candidates, axis=0)
        
        # Tensors
        query_tensor = torch.FloatTensor(query_emb).unsqueeze(0)  # (1, embed_dim)
        candidate_tensor = torch.FloatTensor(candidate_array)  # (num_candidates, embed_dim)
        
        # Project query and candidates
        Q = self.query_proj(query_tensor)  # (1, hidden_dim)
        K = self.key_proj(candidate_tensor)  # (num_candidates, hidden_dim)
        V = self.value_proj(candidate_tensor)  # (num_candidates, hidden_dim)
        
        # Multi-head attention setup
        Q = Q.view(1, self.num_heads, self.head_dim)  # (1, num_heads, head_dim)
        K = K.view(num_candidates, self.num_heads, self.head_dim)  # (num_candidates, num_heads, head_dim)
        V = V.view(num_candidates, self.num_heads, self.head_dim)  # (num_candidates, num_heads, head_dim)
        
        Q = Q.transpose(0, 1)  # (num_heads, 1, head_dim)
        K = K.transpose(0, 1)  # (num_heads, num_candidates, head_dim)
        V = V.transpose(0, 1)  # (num_heads, num_candidates, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (num_heads, 1, num_candidates)

        if relation_type is not None:
            rel_tensor = torch.LongTensor(relation_type)  # (num_candidates,)
            rel_emb = self.relation_embed(rel_tensor)  # (num_candidates, hidden_dim)
            rel_emb = rel_emb.view(num_candidates, self.num_heads, self.head_dim)  # (num_candidates, num_heads, head_dim)
            rel_emb = rel_emb.transpose(0, 1)  # (num_heads, num_candidates, head_dim)
            
            rel_scores = torch.matmul(Q, rel_emb.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (num_heads, 1, num_candidates)
            scores = scores + 0.3 * rel_scores
        
        attn_weights = F.softmax(scores, dim=-1)  # (num_heads, 1, num_candidates)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (num_heads, 1, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous()  # (1, num_heads, head_dim)
        attn_output = attn_output.view(1, -1)  # (1, hidden_dim)
        
        candidate_repr = V.transpose(0, 1).contiguous().view(num_candidates, -1)  # (num_candidates, hidden_dim)
        
        # Broadcast query representation to match candidates
        query_repr = attn_output.expand(num_candidates, -1)  # (num_candidates, hidden_dim)
        
        # Combine query and candidate representations
        combined = self.layer_norm(query_repr + candidate_repr)  # (num_candidates, hidden_dim)
        
        # Project to scores
        scores = self.out_proj(combined).squeeze()  # (num_candidates,)
        
        # Handle single candidate case
        if num_candidates == 1:
            scores = scores.unsqueeze(0)
        
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

    def encode_community_context(self, community_id, community_history, community_sizes): 
        if community_id is None:
            return torch.zeros(256)
        
        visit_count = community_history.count(community_id)
        size = community_sizes.get(community_id, 100)
        
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
        
        # FIX: Validate candidate embeddings
        normalized_candidates = []
        for i, emb in enumerate(candidate_embs):
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            emb = np.array(emb).flatten()
            if emb.shape[0] != self.embed_dim:
                if emb.shape[0] < self.embed_dim:
                    emb = np.pad(emb, (0, self.embed_dim - emb.shape[0]))
                else:
                    emb = emb[:self.embed_dim]
            normalized_candidates.append(emb)
        
        query_tensor = torch.FloatTensor(query_emb)
        candidate_tensor = torch.FloatTensor(np.stack(normalized_candidates, axis=0))
        
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
