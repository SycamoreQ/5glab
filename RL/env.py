import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from utils.coldstart import ColdStartHandler
from utils.diversity import DiversitySelector
from utils.userfeedback import UserFeedbackTracker
from model.llm.parser.unified import ParserType , UnifiedQueryParser , QueryRewardCalculator
from utils.attention_selector import HybridAttentionSelector
from RL.curiousity import CuriosityModule
import random
from model.llm.parser.dspy_parser import DSPyHierarchicalParser , QueryIntent , HierarchicalRewardMapper

try:
    from graph.database.comm_det import CommunityDetector
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False
    print("⚠ Community detection not available")


class RelationType:
    CITES = 0
    CITED_BY = 1
    WROTE = 2
    AUTHORED = 3
    SELF = 4
    COLLAB = 5
    KEYWORD_JUMP = 6
    VENUE_JUMP = 7
    OLDER_REF = 8
    NEWER_CITED_BY = 9
    SECOND_COLLAB = 10
    STOP = 11
    INFLUENCE_PATH = 12


class CommunityAwareRewardConfig:
    SEMANTIC_WEIGHT = 10.0  
    GOAL_REACHED_BONUS = 50.0  
    PROGRESS_REWARD = 5.0 
    STAGNATION_PENALTY = -1.5  
    
    INTENT_MATCH_REWARD = 1.0  
    INTENT_MISMATCH_PENALTY = -0.2 
    
    DIVERSITY_BONUS = 1.0  
    NOVELTY_BONUS = 3.0  
    MAX_NOVELTY_BONUS = 8.0  
    HIGH_DEGREE_BONUS = 0.5 
    
    DEAD_END_PENALTY = -0.3 
    REVISIT_PENALTY = -0.2 
    
    COMMUNITY_SWITCH_BONUS = 2.0 
    COMMUNITY_STUCK_PENALTY = -0.5  
    COMMUNITY_LOOP_PENALTY = -0.3 
    DIVERSE_COMMUNITY_BONUS = 2.0 
    COMMUNITY_SIZE_BONUS = 0.2  
    
    TEMPORAL_JUMP_BONUS = 0.4 
    TEMPORAL_JUMP_PENALTY = -0.05 
    
    STUCK_THRESHOLD = 4 
    SEVERE_STUCK_THRESHOLD = 7  
    SEVERE_STUCK_MULTIPLIER = 1.5  

    AUTHOR_MATCH_BONUS = 15.0  
    AUTHOR_COLLAB_BONUS = 5.0 
    PROLIFIC_AUTHOR_BONUS = 2.0  
    PROLIFIC_THRESHOLD = 20  
    COLLABORATION_BONUS = 1.5 
    COLLABORATION_THRESHOLD = 10
    AUTHOR_H_INDEX_BONUS = 2.0  
    H_INDEX_THRESHOLD = 20 
    
    CITATION_COUNT_BONUS = 0.05  
    RECENCY_BONUS = 0.3  
    CITATION_VELOCITY_BONUS = 0.2  
    INSTITUTION_QUALITY_BONUS = 1.0  
    
    NODE_TYPE_SWITCH = 0.5
    
    TOP_VENUES = {"Nature", "Science", "Cell", "PNAS", "CVPR", "ICCV", 
                  "ECCV", "NeurIPS", "ICML", "ICLR", "ACL", "EMNLP"}
    TOP_INSTITUTIONS = {"MIT", "Stanford", "Berkeley", "CMU", "Harvard",
                        "Oxford", "Cambridge", "ETH", "Google", "DeepMind"}



class AdvancedGraphTraversalEnv:
    """
    Community-aware hierarchical RL environment.
    Tracks and penalizes agent for getting stuck in local graph communities.
    """
    
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2", 
                use_communities=True , use_feedback = True , use_query_parser = True , parser_type: str = 'rule' , 
                use_manager_policy:bool = True , precomputed_embeddings: Dict = None , **kwargs):
        self.store = store
        self.encoder = SentenceTransformer(embedding_model_name)
        self.query_embedding = None
        self.current_intent = None
        self.current_node = None
        self.visited = set()
        self.max_steps = 50
        self.current_step = 0
        self.text_dim = 384
        self.intent_dim = 5
        self.state_dim = self.text_dim * 2 + self.intent_dim
        self.training_edge_cache = None
        self.require_precomputed_embeddings = bool(kwargs.get("require_precomputed_embeddings", True))

        
        self.pending_manager_action = None
        self.available_worker_nodes = []
        
        self.config = CommunityAwareRewardConfig()
        self.trajectory_history = []
        self.relation_types_used = set()
        self.best_similarity_so_far = -1.0
        self.previous_node_embedding = None
        self.previous_node_type = None
        
        self.use_communities = use_communities and COMMUNITY_AVAILABLE
        self.use_authors = use_communities and COMMUNITY_AVAILABLE
        self.community_detector = None

        self.max_revisits = 3 
        self.node_visit_count = {}  # Track visit count per node
        #self.revisit_budget = 0.3

        #self.episode_count = 0
        #self.recent_success_rate = 0.0
        #self.recent_avg_steps = 0.0
     
        
        if self.use_communities and self.use_authors:
            self.current_comm_node = None
            self.prev_comm_node = None
            self.community_detector = CommunityDetector(cache_file= 'community_cache_1M.pkl')
            if not self.community_detector.load_cache():
                print("No community cache found")
                print("Continuing without community rewards")
                self.use_communities = False
            else:
                num_papers = len(self.community_detector.paper_communities)
                num_authors = len(self.community_detector.author_communities)

                if num_papers == 0:
                    print(f"Cache loaded but empty! Papers: {num_papers}")
                    self.use_communities = False
                else:
                    print(f"Community cache validated: {num_papers:,} papers, {num_authors:,} authors")
        
        
        self.current_community = None
        self.community_history = []  
        self.community_visit_count = Counter()  
        self.comm_influence = {}
        self.steps_in_current_community = 0
        self.previous_community = None
        self.visited_communities = set()

        self.current_author = None 
        self.author_hist = []
        self.author_visit_count = Counter()
        self.author_influence = {}
        self.previous_author = None 
        self.h_index_cache = {}

        self.use_feedback = use_feedback
        if use_feedback:
            self.feedback_tracker = UserFeedbackTracker()
            self.diversity_selector = DiversitySelector()
            self.cold_start_handler = ColdStartHandler(store, self.config)

        self.episode_stats = {
            'total_nodes_explored': 0,
            'unique_relation_types': 0,
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_achieved': 0.0,
            'unique_communities_visited': 0,
            'community_switches': 0,
            'max_steps_in_community': 0,
            'community_loops': 0
        }

        self.author_stats = {
            'total_authors_explored': 0, 
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_acheived': 0,
            'unique_communitites_visited': 0,
            'community_switches': 0,
            'max_steps_in_community':0,
            'community_loops': 0, 
            'max_h_index': 0, 
        }

        parser_type_enum = {
            'dspy': ParserType.DSPY,
            'llm': ParserType.LLM_MANUAL,
            'rule': ParserType.RULE_BASED
        }.get(parser_type, ParserType.LLM_MANUAL)
        
        self.use_query_parser = use_query_parser
        self.query_parser = UnifiedQueryParser(
            primary_parser=parser_type_enum,
            model="llama3.2",
            fallback_on_error=True 
        )

        if use_query_parser and parser_type == "dspy":
            self.query_parser = DSPyHierarchicalParser(
                model="llama3.2",
                optimize=False 
            )
            self.reward_mapper = HierarchicalRewardMapper(self.config)
            self.query_intent = None
            print("[ENV] Using DSPy hierarchical parser")
        else:
            self.query_parser = None
            self.reward_mapper = None
            self.query_intent = None

        self.hierarchical_reward_mapper = HierarchicalRewardMapper(self.config)
        self.query_intent: Optional[QueryIntent] = None

        if use_manager_policy: 
            from RL.manager_policy import AdaptiveManagerPolicy
            self.manager_policy = AdaptiveManagerPolicy(state_dim=self.state_dim)
        else:
            self.use_manager_policy = False

        self.precomputed_embeddings = precomputed_embeddings or {}
        if torch.cuda.is_available():
            from utils.batchencoder import BatchEncoder
            self.encoder = BatchEncoder()
            print("Using GPU-accelerated encoder")

        self.attention_selector = HybridAttentionSelector(
        embed_dim=self.text_dim,
        use_community_attention=self.use_communities
        )
        self.use_attention = True

        community_feature_dim = 10 if self.use_communities else 0
        actual_state_dim = self.text_dim * 2 + self.intent_dim + community_feature_dim
        print(f"[ENV] State dimension: {actual_state_dim} (query:{self.text_dim} + node:{self.text_dim} + intent:{self.intent_dim} + community:{community_feature_dim})")

        self.curiosity_module = CuriosityModule(
            state_dim=actual_state_dim, 
            embedding_dim=self.text_dim 
        )

    def _normalize_node_keys(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes Neo4j query results to consistent format."""
        if not node:
            return {}
        
        normalized = {}
        for key, value in node.items():
            clean_key = key.split('.')[-1] if '.' in key else key
            normalized[clean_key] = value
        
        if 'fields' in normalized and 'fieldsOfStudy' not in normalized:
            normalized['fieldsOfStudy'] = normalized['fields']
        if 'venue' not in normalized and 'publication_name' in normalized:
            normalized['venue'] = normalized['publication_name']

        if 'paperId' in normalized and 'paper_id' not in normalized:
            normalized['paper_id'] = str(normalized['paperId'])
        if 'paper_id' in normalized and 'paperId' not in normalized:
            normalized['paperId'] = str(normalized['paper_id'])

        if 'authorId' in normalized and 'author_id' not in normalized:
            normalized['author_id'] = str(normalized['authorId'])
        if 'author_id' in normalized and 'authorId' not in normalized:
            normalized['authorId'] = str(normalized['author_id'])
        
        return normalized


    def _get_intent_vector(self, intent_enum: int):
        vec = np.zeros(self.intent_dim)
        if intent_enum < self.intent_dim:
            vec[intent_enum] = 1.0
        return vec
    

    def _update_community_tracking(self, node_id: str):
        if not self.use_communities:
            return
        
        #if len(self.community_history) == 0:
        #    print(f"[DEBUG] First community lookup for node: {node_id}")


        new_community = self.community_detector.get_community(node_id)

        #if len(self.community_history) == 0:
        #    print(f"[DEBUG] Community result: {new_community}")
        #    if new_community is None:
        #        print(f"[DEBUG] Node ID format: {node_id}")
        #        sample = list(self.community_detector.paper_communities.keys())[:3]
        #        print(f"[DEBUG] Sample cache IDs: {sample}")
        
        if new_community is None:
            return
        
        if new_community != self.current_community:
            self.episode_stats['community_switches'] += 1
            
            if new_community in self.visited_communities:
                self.episode_stats['community_loops'] += 1
            else:
                self.visited_communities.add(new_community)
            
            self.previous_community = self.current_community
            self.current_community = new_community
            self.steps_in_current_community = 1
        else:
            self.steps_in_current_community += 1
            self.episode_stats['max_steps_in_community'] = max(
                self.episode_stats['max_steps_in_community'],
                self.steps_in_current_community
            )
        
        self.community_history.append(new_community)
        self.episode_stats['unique_communities_visited'] = len(self.visited_communities)

    def _normalize_paper_id(self, paper_id: str) -> str:
        """Normalize paper ID to consistent format."""
        if not paper_id:
            return ""

        paper_id = str(paper_id).strip()
        paper_id = paper_id.lstrip('0')
        return paper_id if paper_id else "0"
    

    def calculate_community_reward(self) -> Tuple[float, str, str]:
        if not self.use_communities or self.current_community is None:
            return (0.0, None, 'no_community')
        
        paper_reward = 0.0
        reasons = []
        
        paper_id = self.current_comm_node.get('paperId')

        if not self.use_communities: 
            return 0.0 , None , 'no_community'
        
        if self.previous_community and self.previous_community.startswith('MISC_P'):
            if self.current_community and not self.current_community.startswith('MISC_P'):
                paper_reward += 3.0  
                reasons.append("escaped_MISC:+3.0")
        if (self.steps_in_current_community == 1 and 
            self.previous_community and 
            self.current_community != self.previous_community):
            
            paper_reward += self.config.COMMUNITY_SWITCH_BONUS
            reasons.append(f'switch:+{self.config.COMMUNITY_SWITCH_BONUS:.1f}')
            
            if self.current_community not in self.visited_communities:
                discovery_bonus = 3.0
                paper_reward += discovery_bonus
                reasons.append(f'discovery:+{discovery_bonus:.1f}')
        

        if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
            base_penalty = self.config.COMMUNITY_STUCK_PENALTY
            
            if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                multiplier = 1.5 ** (self.steps_in_current_community - self.config.SEVERE_STUCK_THRESHOLD)
                penalty = base_penalty * min(multiplier, 5.0)  
                paper_reward += penalty
                reasons.append(f'severe_stuck:{penalty:.1f}')
            else:
                paper_reward += base_penalty
                reasons.append(f'stuck:{base_penalty:.1f}')

        if len(self.community_history) > 2:
            previous_communities = [c for c in self.community_history[:-1]]
            if self.current_community in previous_communities:
                loop_penalty = self.config.COMMUNITY_LOOP_PENALTY
                
                recency = len(self.community_history) - previous_communities.index(self.current_community) - 1
                if recency <= 2:  
                    loop_penalty *= 2.0
                
                paper_reward += loop_penalty
                reasons.append(f'loop:{loop_penalty:.1f}')
        
        unique_communities = len(self.visited_communities)
        if unique_communities >= 5:
            diversity_bonus = min((unique_communities - 4) * 1.0, 10.0)
            paper_reward += diversity_bonus
            reasons.append(f'diversity:+{diversity_bonus:.1f}')
        
        if paper_id:
            size = self.community_detector.get_community_size(self.current_community)
            if 10 <= size <= 100: 
                size_bonus = self.config.COMMUNITY_SIZE_BONUS
                paper_reward += size_bonus
                reasons.append(f'size:+{size_bonus:.1f}')
            elif size < 5:  
                paper_reward -= 0.5
                reasons.append('tiny:-0.5')
        
        if len(self.community_history) > 1:
            try:
                prev_comm = self.community_history[-2]
                prev_year = int(prev_comm.split('_')[-1])
                curr_year = int(self.current_community.split('_')[-1])
                
                if curr_year > prev_year:  
                    paper_reward += self.config.TEMPORAL_JUMP_BONUS
                    reasons.append(f'temporal:+{self.config.TEMPORAL_JUMP_BONUS:.1f}')
                elif curr_year < prev_year - 3: 
                    paper_reward += self.config.TEMPORAL_JUMP_PENALTY
                    reasons.append(f'old_jump:{self.config.TEMPORAL_JUMP_PENALTY:.1f}')
            except:
                pass
        
        reason_str = ', '.join(reasons) if reasons else 'none'
        return (paper_reward, 'Paper', reason_str)
    

    async def calculate_author_reward(self) -> Tuple[float , str , str]: 
        if not self.current_node: 
            return 0.0 , None , 'no_node'
        
        reward = 0.0
        author_id = self.current_node.get('authorId') or self.current_node.get('authorId')

        if not author_id or not self.use_authors: 
            return 0.0 , None , 'no_author' 
        
        reasons = []

        if self.query_intent and hasattr(self.query_intent , "target_entity"):
            target = self.query_intent.target_entity.lower()
            author_name = self.current_node.get(('name' , '') or '').lower()

            if target in author_name or author_name in target:
                reward += self.config.AUTHOR_MATCH_BONUS
                reasons.append(f"target_match+{self.config.AUTHOR_MATCH_BONUS:.1f}")
            elif len(self.trajectory_history) > 0:
                for traj in self.trajectory_history[-3:]: 
                    prev_node = traj['node']
                    if prev_node.get('authorId'):
                        reward += self.config.AUTHOR_COLLAB_BONUS
                        reasons.append(f"collab+{self.config.AUTHOR_COLLAB_BONUS:.1f}")
                        break


        paper_count = await self.store.get_paper_by_author_id(author_id)
        if paper_count >= self.config.PROLIFIC_THRESHOLD:
            prolific_bonus = min(paper_count / 50.0, 5.0)  
            reward += prolific_bonus
            reasons.append(f"prolific+{prolific_bonus:.1f}")


        h_index = self.current_node.get('hIndex', 0) or self.current_node.get('hindex', 0)
        if h_index >= self.config.H_INDEX_THRESHOLD:
            h_bonus = min(np.log1p(h_index - 20) * self.config.AUTHOR_H_INDEX_BONUS, 5.0)
            reward += h_bonus
            reasons.append(f"h_index+{h_bonus:.1f}")

            self.author_stats['max_hindex'] = max(self.author_stats.get('max_hindex', 0), h_index)

        if hasattr(self.store, 'get_collab_count_by_author'):
            collab_count = await self.store.get_collab_count_by_author(author_id)
            if collab_count >= self.config.COLLABORATION_THRESHOLD:
                collab_bonus = min(collab_count / 20.0, 3.0)
                reward += collab_bonus
                reasons.append(f"collabs+{collab_bonus:.1f}")

        if len(self.trajectory_history) > 0:
            prev_was_paper = 'paperId' in self.trajectory_history[-1]['node']
            if prev_was_paper:
                reward += self.config.NODE_TYPE_SWITCH
                reasons.append(f"switch+{self.config.NODE_TYPE_SWITCH:.1f}")
        

        reason_str = ', '.join(reasons) if reasons else 'none'
        return reward, 'Author', reason_str



    def _calculate_trajectory_rewards(self) -> float:
        if len(self.trajectory_history) < 2 :
            return 0.0

        visited_embeddings = []
        for traj in self.trajectory_history: 
            node = traj['node']
            paper_id = node.get('paper_id')
            if paper_id and paper_id in self.precomputed_embeddings: 
                visited_embeddings.append(self.precomputed_embeddings[paper_id])

        if len(visited_embeddings) < 2: 
            return 0.0 


        embeddings = np.array(visited_embeddings)
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(embeddings)

        n = len(sim)
        avg_sim = (sim.sum()-n)/(n*(n - 1))
        div = 1 - avg_sim
        div_reward = max(0 , div*20)
        return div_reward 
                 

    async def reset(self, query: str, intent: int, start_node_id: str = None , force_mode: Optional[str] = None) -> Tuple[np.ndarray , Dict]:
        if self.query_parser:
            self.query_intent = self.query_parser.parse(query)
            
            if self.current_step <= 1: 
                print(f"\n[QUERY] {query}")
                print(f"[PARSE] Target: {self.query_intent.target_entity} | Op: {self.query_intent.operation}")
                print(f"[PARSE] Semantic: {self.query_intent.semantic}")
                if self.query_intent.constraints:
                    print(f"[PARSE] Constraints: {len(self.query_intent.constraints)}")
                    for c in self.query_intent.constraints:
                        print(f" - {c}")
        else: 
            self.query_facets = {'semantic' : query , 'original' : query}

        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
        self.node_visit_count = {}
        self.current_step = 0
        self.pending_manager_action = None
        self.available_worker_nodes = []
        self.trajectory_history = []
        self.relation_types_used = set()
        self.best_similarity_so_far = -1.0
        self.previous_node_embedding = None

        self.current_community = None
        self.community_history = []
        self.community_visit_count = Counter()
        self.steps_in_current_community = 0
        self.previous_community = None
        self.visited_communities = set()
        
        self.episode_stats = {
            'total_nodes_explored': 0,
            'unique_relation_types': 0,
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_achieved': 0.0,
            'unique_communities_visited': 0,
            'community_switches': 0,
            'max_steps_in_community': 0,
            'community_loops': 0
        }

        
        if start_node_id:
            node = await self.store.get_paper_by_id(start_node_id)
            if not node:
                raise ValueError(f"Paper with ID {start_node_id} not found.")
            self.current_node = self._normalize_node_keys(node)
        else:
            keywords = query.lower().split()
            candidates = []
            
            for keyword in keywords:
                if len(keyword) < 3:
                    continue
                results = await self.store.get_paper_by_title(keyword)
                if results:
                    candidates.extend(results[:5])
            
            if not candidates:
                field_query = """
                MATCH (p:Paper)
                WHERE 'Computer Science' IN p.fieldsOfStudy
                AND p.citationCount > 20
                AND p.year > 2015
                AND p.abstract IS NOT NULL
                RETURN p
                ORDER BY p.citationCount DESC
                LIMIT 20
                """
                candidates = await self.store._run_query_method(field_query)
            
            if not candidates:
                raise ValueError(f"Could not find ANY starting papers for query: {query}")
            
            self.current_node = self._normalize_node_keys(np.random.choice(candidates))
        
        if not self.current_node:
            raise ValueError(f"Failed to initialize current_node")
        
        paper_id = self.current_node.get('paper_id')
        if not paper_id:
            raise ValueError(f"Node has no paper_id")
        
        self.visited.add(paper_id)
        self._update_community_tracking(paper_id)
        self.previous_node_embedding = await self._get_node_embedding(self.current_node)
        
        return await self._get_state()
    

    async def reset_with_cache_validation(self, query: str, intent: int, start_node_id: str = None):
        """
        Improved reset with cache validation and semantic starting node selection.
        """
        if not hasattr(self, 'training_paper_ids'):
            if self.precomputed_embeddings:
                self.training_paper_ids = set(self.precomputed_embeddings.keys())
                print(f"[INIT] Auto-initialized training_paper_ids with {len(self.training_paper_ids):,} papers from embeddings")
            else:
                self.training_paper_ids = set()
                print("[WARN] No training_paper_ids or precomputed_embeddings available")
        
        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
        self.node_visit_count = {}
        self.current_step = 0
        self.pending_manager_action = None
        self.available_worker_nodes = []
        self.trajectory_history = []
        self.relation_types_used = set()
        self.best_similarity_so_far = -1.0
        self.previous_node_embedding = None
        
        # Reset community tracking
        self.current_community = None
        self.community_history = []
        self.community_visit_count = Counter()
        self.steps_in_current_community = 0
        self.previous_community = None
        self.visited_communities = set()
        
        self.episode_stats = {
            'total_nodes_explored': 0,
            'unique_relation_types': 0,
            'dead_ends_hit': 0,
            'revisits': 0,
            'max_similarity_achieved': 0.0,
            'unique_communities_visited': 0,
            'community_switches': 0,
            'max_steps_in_community': 0,
            'community_loops': 0
        }
        
        self.current_query_facets = self.query_parser.parse(query)
        
        if start_node_id:
            node = await self.store.get_paper_by_id(start_node_id)
            if not node:
                raise ValueError(f"Paper with ID {start_node_id} not found.")
            self.current_node = self._normalize_node_keys(node)
        else:
            candidates = []
            
            if hasattr(self, 'training_paper_ids') and len(self.training_paper_ids) > 0:
                sample_size = min(500, len(self.training_paper_ids))
                sample_ids = random.sample(list(self.training_paper_ids), sample_size)

                for pid in sample_ids:
                    if pid in self.precomputed_embeddings:
                        paper = await self.store.get_paper_by_id(pid)
                        if paper:
                            candidates.append(paper)
            
            if not candidates and self.precomputed_embeddings:
                print("[INIT] Using precomputed_embeddings for candidate selection")
                sample_pids = list(self.precomputed_embeddings.keys())[:1000]
                for pid in sample_pids:
                    paper = await self.store.get_paper_by_id(pid)
                    if paper:
                        candidates.append(paper)
                        if len(candidates) >= 100:
                            break
            
            if not candidates:
                print("[INIT] Falling back to keyword matching")
                keywords = query.lower().split()
                for keyword in keywords:
                    if len(keyword) < 3:
                        continue
                    results = await self.store.get_paper_by_title(keyword)
                    candidates.extend(results[:20])
                    if len(candidates) >= 50:
                        break
            
            if not candidates:
                raise ValueError(f"Could not find ANY starting papers for query: {query}")
            
            best_sim = -1.0
            best_paper = None
            
            for paper in candidates:
                paper_emb = await self._get_node_embedding(self._normalize_node_keys(paper))
                if paper_emb is not None and isinstance(paper_emb, np.ndarray):
                    sim = np.dot(self.query_embedding, paper_emb) / (
                        np.linalg.norm(self.query_embedding) * np.linalg.norm(paper_emb) + 1e-9
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_paper = paper
            
            if best_paper is None:
                best_paper = np.random.choice(candidates)
                print(f"[INIT] Random starting paper (no embeddings matched)")
            else:
                print(f"[INIT] Starting similarity: {best_sim:.3f}")
            
            self.current_node = self._normalize_node_keys(best_paper)
        
        if not self.current_node:
            raise ValueError(f"Failed to initialize current_node")
        
        paper_id = self.current_node.get('paper_id')
        if not paper_id:
            raise ValueError(f"Node has no paper_id")
        
        self.visited.add(paper_id)
        self._update_community_tracking(paper_id)
        self.previous_node_embedding = await self._get_node_embedding(self.current_node)
        
        return await self._get_state()



    async def _get_node_embedding(self, node: Dict[str, Any]) -> np.ndarray:
        """Get embedding for a node. ALWAYS returns valid ndarray, never None."""
        if not node:
            logging.warning("Empty node passed to function")
            return np.zeros(self.text_dim, dtype=np.float32)
        
        paper_id = node.get('paper_id') or node.get('paperId')
        author_id = node.get('author_id') or node.get('authorId')

        
        if paper_id and paper_id in self.precomputed_embeddings:
            return self.precomputed_embeddings[paper_id]
        if author_id and author_id in self.precomputed_embeddings: 
            return self.precomputed_embeddings[author_id]
        
        node_text = None 
        
        if paper_id: 
        
            title = str(node.get('title', ''))[:200] if node.get('title') else ''
            abstract = str(node.get('abstract', ''))[:1500] if node.get('abstract') else ''
            
            INVALID_TITLES = {'', 'NA', '...', 'Unknown', 'null', 'None'}
            
            if title in INVALID_TITLES or len(title) < 3:

                fields = node.get('fieldsOfStudy', []) or node.get('fields', [])
                venue = node.get('venue', '') or node.get('publication_name', '')
                year = node.get('year', '')
                meta = []

                if fields:
                    meta.append("fields" + ",".join(str(fields[:5])))

                if venue: 
                    meta.append("venue" + str(venue))
                
                if year: 
                    meta.append("year" + str(year))
                    
                else:
                    cc = node.get('citationCount') or node.get('citation_count') or ''

                    fallback_parts = [f"Research paper {paper_id}"]
                    if cc:
                        fallback_parts.append(f"citations {cc}")

                    node_text = ", ".join(str(x) for x in fallback_parts if x) 

                parts = [title]
                if abstract and len(abstract) > 20:
                    parts.append(abstract[:500])
                node_text = ' '.join(parts)
            
                
        elif author_id: 
            name = str(node.get('name' , ''))[:50] if node.get('name') else ''
            
            INVALID_NAMES = {'' , 'NA' , '...' , 'Unknown' , 'NULL' , 'null' , 'None' , 'undefined' , 'XYZ'}
            if name and name not in INVALID_NAMES and len(name) > 1: 
                paper_count = node.get('paper_count', 0)
                h_index = node.get('h_index', 0)
                
                if paper_count or h_index:
                    node_text = f"{name}, {paper_count} papers, h-index {h_index}"
            
                else:
                    node_text = f"Author: {name}"
        
            if not node_text: 
                node_text = f"Author {author_id}"

        if not node_text or len(node_text) < 5:
            if paper_id:
                node_text = f"Research paper {paper_id[-10:]}"
            else:
                if self.current_step <= 3:
                    logging.warning(f"Node has no usable text: {paper_id}")
                return np.zeros(self.text_dim, dtype=np.float32)
        
            
        try:
            if hasattr(self.encoder, 'encode_with_cache'):
                embedding = self.encoder.encode_with_cache(node_text, cache_key=paper_id)
            else:
                embedding = self.encoder.encode(node_text)
            
            if embedding is None:
                logging.warning(f"Encoder returned None for text: {node_text[:50]}")
                return np.zeros(self.text_dim, dtype=np.float32)
            
            if not isinstance(embedding, np.ndarray):
                logging.warning(f"Encoder returned non-ndarray: {type(embedding)}")
                return np.zeros(self.text_dim, dtype=np.float32)
            
            return embedding
        except Exception as e:
            logging.error(f"Encoding exception: {e}. Node: {paper_id}")
            return np.zeros(self.text_dim, dtype=np.float32)



    async def _get_state(self):
        """Constructs state vector."""
        node_emb = await self._get_node_embedding(self.current_node)
        intent_vec = self._get_intent_vector(self.current_intent)
        community_features = self._get_community_features()
    
        return np.concatenate([
            self.query_embedding,      
            node_emb,                
            intent_vec,              
            community_features         
        ]) 


    def _get_community_features(self) -> np.ndarray:
        """Extract community-aware features for agent state."""
        features = np.zeros(10, dtype=np.float32)
        
        if not self.use_communities or self.current_community is None:
            return features
        
        features[0] = min(self.steps_in_current_community / 10.0, 1.0)
    
        features[1] = min(len(self.visited_communities) / 20.0, 1.0)
        
        comm_size = self.community_detector.get_community_size(self.current_community)
        features[2] = min(np.log1p(comm_size) / 5.0, 1.0)  
        features[3] = 1.0 if (self.previous_community and 
                            self.current_community != self.previous_community) else 0.0
    
        if len(self.community_history) > 1:
            previous_comms = [c for c in self.community_history[:-1]]
            features[4] = 1.0 if self.current_community in previous_comms else 0.0

        features[5] = 1.0 if self.steps_in_current_community >= self.config.STUCK_THRESHOLD else 0.0
        
        if self.available_worker_nodes:
            target_communities = set()
            for node, _ in self.available_worker_nodes:
                node_id = node.get('paperId')
                if node_id:
                    comm = self.community_detector.get_community(node_id)
                    if comm:
                        target_communities.add(comm)
            features[6] = min(len(target_communities) / 5.0, 1.0)
        
        
        visit_count = self.community_visit_count.get(self.current_community, 0)
        features[8] = min(visit_count / 5.0, 1.0)
        
        # Feature 9: Exploration pressure (stuck + loop combined)
        if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
            features[9] = 1.0  # Strong signal to leave
        elif self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
            features[9] = 0.5
        
        return features 


    async def get_manager_actions(self) -> List[int]:
        """Manager decides which relation type to explore."""
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        valid_relations = []
        
        if paper_id:
            refs = await self.store.get_references_by_paper(paper_id)
            if refs and any(not self._is_visited(self._normalize_node_keys(r)) for r in refs):
                valid_relations.append(RelationType.CITES)
            
            # 2. CITED
            cites = await self.store.get_citations_by_paper(paper_id)
            if cites and any(not self._is_visited(self._normalize_node_keys(c)) for c in cites):
                valid_relations.append(RelationType.CITED_BY)
            
            # 3. WROTE
            authors = await self.store.get_authors_by_paper_id(paper_id)
            if authors and any(not self._is_visited(self._normalize_node_keys(a)) for a in authors):
                valid_relations.append(RelationType.WROTE)
            
            # 4. KEYWORD JUMP
            keywords = self.current_node.get('fieldsOfStudy') or self.current_node.get('fields')
            if keywords: 
                valid_relations.append(RelationType.KEYWORD_JUMP)
            
            # 5. VENUE JUMP
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue: 
                valid_relations.append(RelationType.VENUE_JUMP)
            
            # 6. OLDER/NEWER 
            older = await self.store.get_older_references(paper_id)
            if older and any(not self._is_visited(self._normalize_node_keys(o)) for o in older):
                valid_relations.append(RelationType.OLDER_REF)
            
            newer = await self.store.get_newer_citations(paper_id)
            if newer and any(not self._is_visited(self._normalize_node_keys(n)) for n in newer):
                valid_relations.append(RelationType.NEWER_CITED_BY)
        
        elif author_id:
            papers = await self.store.get_papers_by_author_id(author_id)
            if papers and any(not self._is_visited(self._normalize_node_keys(p)) for p in papers):
                valid_relations.append(RelationType.AUTHORED)
            
            collabs = await self.store.get_collabs_by_author(author_id)
            if collabs and any(not self._is_visited(self._normalize_node_keys(c)) for c in collabs):
                valid_relations.append(RelationType.COLLAB)
            
            second_collabs = await self.store.get_second_degree_collaborators(author_id , limit = 20)
            if second_collabs and any(not self._is_visited(self._normalize_node_keys(c)) for c in second_collabs):
                valid_relations.append(RelationType.SECOND_COLLAB)
            
            influence = await self.store.get_influence_path_papers(author_id , limit = 10)
            if influence and any(not self._is_visited(self._normalize_node_keys(i)) for i in influence):
                valid_relations.append(RelationType.INFLUENCE_PATH)
        
        valid_relations.append(RelationType.STOP)
        
        return valid_relations
    

    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        self.pending_manager_action = relation_type
        
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps
            if self.previous_node_embedding is None:
                return True, -1.0
            
            current_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            
            if current_sim > 0.7:
                return True, self.config.GOAL_REACHED_BONUS
            elif current_sim > 0.4:
                return True, 0.5
            else:
                return True, -1.0
        
        manager_reward = 0.0
        paper_id = str(self.current_node.get('paperId') or self.current_node.get('paper_id') or '')
        author_id = str(self.current_node.get('authorId') or self.current_node.get('author_id') or '')
        
        raw_nodes = []
    
        if hasattr(self, 'training_edge_cache') and self.training_edge_cache and paper_id:
            edges = self.training_edge_cache.get(paper_id, [])
            
            if relation_type == RelationType.CITES:
                target_ids = [tid for et, tid in edges if et == "cites"]
                for tid in target_ids:
                    if tid in self.precomputed_embeddings:
                        paper = await self.store.get_paper_by_id(tid)
                        if paper:
                            raw_nodes.append(paper)
            
            elif relation_type == RelationType.CITED_BY:
                target_ids = [tid for et, tid in edges if et == "citedby"]
                for tid in target_ids:
                    if tid in self.precomputed_embeddings:
                        paper = await self.store.get_paper_by_id(tid)
                        if paper:
                            raw_nodes.append(paper)
        
        else:
            if relation_type == RelationType.CITES and paper_id:
                raw_nodes = await self.store.get_references_by_paper(paper_id)
            elif relation_type == RelationType.CITED_BY and paper_id:
                raw_nodes = await self.store.get_citations_by_paper(paper_id)
            elif relation_type == RelationType.WROTE and paper_id:
                raw_nodes = await self.store.get_authors_by_paper_id(paper_id)
            elif relation_type == RelationType.AUTHORED and author_id:
                raw_nodes = await self.store.get_papers_by_author_id(author_id)
            elif relation_type == RelationType.COLLAB and author_id:
                raw_nodes = await self.store.get_collabs_by_author(author_id)
            elif relation_type == RelationType.KEYWORD_JUMP and paper_id:
                keywords = self.current_node.get('keywords')
                if keywords:
                    keyword = keywords.split(',')[0].strip() if isinstance(keywords, str) else str(keywords[0])
                    raw_nodes = await self.store.get_papers_by_keyword(keyword, limit=5, exclude_paper_id=paper_id)
            elif relation_type == RelationType.VENUE_JUMP and paper_id:
                venue = self.current_node.get('publicationName') or self.current_node.get('venue')
                if venue:
                    raw_nodes = await self.store.get_papers_by_venue(venue, exclude_paper_id=paper_id)
            elif relation_type == RelationType.OLDER_REF and paper_id:
                raw_nodes = await self.store.get_older_references(paper_id)
            elif relation_type == RelationType.NEWER_CITED_BY and paper_id:
                raw_nodes = await self.store.get_newer_citations(paper_id)
            elif relation_type == RelationType.SECOND_COLLAB and author_id:
                raw_nodes = await self.store.get_second_degree_collaborators(author_id, limit=20)
            elif relation_type == RelationType.INFLUENCE_PATH and author_id:
                raw_nodes = await self.store.get_influence_path_papers(author_id, limit=10)
            
            if relation_type == RelationType.STOP:
                if self.current_step < 5:
                    return False, -5.0
                return True, 0.0
        
        print(f"  [MANAGER] Relation {relation_type}: fetched {len(raw_nodes)} raw nodes")
        
        # Rest of your manager_step logic...
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]
        
        INVALID_VALUES = ['', 'NA', '...', 'Unknown', 'null', 'None', 'undefined']
        valid_nodes = []
        for node in normalized_nodes:
            title = str(node.get('title', '')) if node.get('title') else ''
            name = str(node.get('name', '')) if node.get('name') else ''
            paper_id = node.get('paperId')
            
            title_valid = title and title not in INVALID_VALUES and len(title.strip()) > 3
            if not title_valid and paper_id:
                title_valid = True
                node['title'] = f"Paper {paper_id[-10:]}"
            
            name_valid = name and name not in INVALID_VALUES and len(name.strip()) > 2
            
            if title_valid or name_valid:
                valid_nodes.append(node)
        
        self.available_worker_nodes = []
        for node in valid_nodes:
            if not self._is_visited(node):
                self.available_worker_nodes.append((node, relation_type))
        
        print(f"  [MANAGER] After filtering: {len(self.available_worker_nodes)} valid unvisited nodes")
        
        num_available = len(self.available_worker_nodes)
        if num_available == 0:
            manager_reward = self.config.DEAD_END_PENALTY
            self.episode_stats['dead_ends_hit'] += 1
        elif num_available >= 10:
            manager_reward = self.config.HIGH_DEGREE_BONUS
        elif num_available >= 5:
            manager_reward = 0.5
        
        return False, manager_reward


    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        """Manager step that uses training cache when available."""
        self.pending_manager_action = relation_type
        
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps
            if self.previous_node_embedding is None:
                return True, -1.0
            
            current_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            
            if current_sim > 0.7:
                return True, self.config.GOAL_REACHED_BONUS
            elif current_sim > 0.4:
                return True, 0.5
            else:
                return True, -1.0
        
        manager_reward = 0.0
        paper_id = self._normalize_paper_id(str(self.current_node.get('paperId') or self.current_node.get('paper_id') or ''))
        author_id = self._normalize_paper_id(str(self.current_node.get('authorId') or self.current_node.get('author_id') or ''))
        
        raw_nodes = []
        
        # Use training cache if available
        if hasattr(self, 'training_edge_cache') and self.training_edge_cache and paper_id:
            edges = self.training_edge_cache.get(paper_id, [])
            
            if relation_type == RelationType.CITES:
                target_ids = [tid for et, tid in edges if et == "cites"]
                for tid in target_ids[:50]:  # Increased from 20
                    tid = self._normalize_paper_id(tid)
                    if tid in self.precomputed_embeddings:
                        paper = await self.store.get_paper_by_id(tid)
                        if paper:
                            raw_nodes.append(paper)
            
            elif relation_type == RelationType.CITED_BY:
                target_ids = [tid for et, tid in edges if et == "citedby"]
                for tid in target_ids[:50]:  # Increased from 20
                    tid = self._normalize_paper_id(tid)
                    if tid in self.precomputed_embeddings:
                        paper = await self.store.get_paper_by_id(tid)
                        if paper:
                            raw_nodes.append(paper)
        else:
            # Fallback to database queries
            if relation_type == RelationType.CITES and paper_id:
                raw_nodes = await self.store.get_references_by_paper(paper_id)
            elif relation_type == RelationType.CITED_BY and paper_id:
                raw_nodes = await self.store.get_citations_by_paper(paper_id)
            elif relation_type == RelationType.WROTE and paper_id:
                raw_nodes = await self.store.get_authors_by_paper_id(paper_id)
        
        print(f"  [MANAGER] Relation {relation_type}: fetched {len(raw_nodes)} raw nodes")
        
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]
        
        valid_nodes = []
        filter_reasons = {"no_id": 0, "no_embedding": 0, "visited": 0, "accepted": 0}
        
        for node in normalized_nodes:
            node_paper_id = self._normalize_paper_id(str(node.get('paperId') or node.get('paper_id') or ''))
            
            if not node_paper_id:
                filter_reasons["no_id"] += 1
                continue
            
            if node_paper_id not in self.precomputed_embeddings:
                filter_reasons["no_embedding"] += 1
                if filter_reasons["no_embedding"] <= 3:  # Only log first few
                    print(f"    [FILTER] Paper {node_paper_id[-8:]} not in embeddings (cache has {len(self.precomputed_embeddings):,} papers)")
                continue
            
            if self._is_visited(node):
                filter_reasons["visited"] += 1
                continue
            
            filter_reasons["accepted"] += 1
            valid_nodes.append(node)
        
        print(f"  [FILTER] Results: {filter_reasons}")
        
        self.available_worker_nodes = []
        for node in valid_nodes:
            self.available_worker_nodes.append((node, relation_type))
        
        print(f"  [MANAGER] After filtering: {len(self.available_worker_nodes)} valid unvisited nodes")
        
        num_available = len(self.available_worker_nodes)
        
        if num_available == 0:
            manager_reward = self.config.DEAD_END_PENALTY
            self.episode_stats['dead_ends_hit'] += 1
        elif num_available >= 10:
            manager_reward = self.config.HIGH_DEGREE_BONUS
        elif num_available >= 5:
            manager_reward = 0.5
        
        return False, manager_reward

    def _is_visited(self, node: Dict[str, Any]) -> bool:
        """Check if node has been visited too many times."""
        paper_id = self._normalize_paper_id(str(node.get('paperId') or node.get('paper_id') or ''))
        author_id = self._normalize_paper_id(str(node.get('authorId') or node.get('author_id') or ''))
        node_id = paper_id or author_id
        
        if not node_id:
            return False
        
        visit_count = self.node_visit_count.get(node_id, 0)
        return visit_count >= self.max_revisits
    
    async def get_worker_actions(self) -> List[Tuple[Dict, int]]:
        if not self.available_worker_nodes:
            return []
        
        actions = self.available_worker_nodes
        
        if not self.use_communities:
            return actions
        
        annotated_actions = []
        for node, relation_type in actions:
            paper_id = node.get('paperId')
            if not paper_id:
                annotated_actions.append((node, relation_type, None, False))
                continue
            
            target_comm = self.community_detector.get_community(paper_id)
            is_new_comm = (target_comm != self.current_community and 
                        target_comm not in self.visited_communities)
            
            annotated_actions.append((node, relation_type, target_comm, is_new_comm))
        
        if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
            new_comm_actions = [(n, r) for n, r, c, is_new in annotated_actions if is_new]
            diff_comm_actions = [(n, r) for n, r, c, is_new in annotated_actions 
                                if not is_new and c != self.current_community]
            same_comm_actions = [(n, r) for n, r, c, is_new in annotated_actions 
                                if c == self.current_community]
            
            prioritized = new_comm_actions + diff_comm_actions + same_comm_actions
            
            print(f"  [COMM-FILTER] Stuck! Prioritizing: " +
                f"{len(new_comm_actions)} new, " +
                f"{len(diff_comm_actions)} different, " +
                f"{len(same_comm_actions)} same")
            
            return prioritized 
        
        return actions 




    async def worker_step(self, chosen_node: Dict) -> Tuple[np.ndarray, float, bool]:
        self.current_step += 1
        self.episode_stats['total_nodes_explored'] += 1
        
        self.current_node = self._normalize_node_keys(chosen_node)
        self.current_comm_node = self.current_node
        
        paper_id = self.current_node.get('paperId')
        author_id = self.current_node.get('authorId')
        
        is_revisit = False
        if paper_id:
            if paper_id in self.visited:
                is_revisit = True
                self.episode_stats['revisits'] += 1
            self.node_visit_count[paper_id] = self.node_visit_count.get(paper_id, 0) + 1
            self.visited.add(paper_id)
            self._update_community_tracking(paper_id)
        
        if author_id:
            if author_id in self.visited:
                is_revisit = True
                self.episode_stats['revisits'] += 1
            self.node_visit_count[author_id] = self.node_visit_count.get(author_id, 0) + 1
            self.visited.add(author_id)
            self._update_community_tracking(author_id)
        
        node_emb = await self._get_node_embedding(self.current_node)
        if node_emb is None or not isinstance(node_emb, np.ndarray):
            node_emb = np.zeros(self.text_dim, dtype=np.float32)
        
        semantic_sim = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )

        worker_reward = 0.0
        
        if semantic_sim >= 0.7:
            worker_reward = 100.0  
        elif semantic_sim >= 0.6:
            worker_reward = 50.0 + 50.0 * ((semantic_sim - 0.6) / 0.1)
        elif semantic_sim >= 0.5:
            worker_reward = 25.0 + 25.0 * ((semantic_sim - 0.5) / 0.1)
        elif semantic_sim >= 0.4:
            worker_reward = 10.0 + 15.0 * ((semantic_sim - 0.4) / 0.1)
        elif semantic_sim >= 0.3:
            worker_reward = 0.0 + 10.0 * ((semantic_sim - 0.3) / 0.1)
        elif semantic_sim >= 0.2:
            worker_reward = -5.0 + 5.0 * ((semantic_sim - 0.2) / 0.1)
        else:
            worker_reward = -10.0 + 5.0 * semantic_sim / 0.2
        
        if self.previous_node_embedding is not None:
            prev_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            progress = semantic_sim - prev_sim
            
            if progress > 0.15:
                worker_reward += self.config.PROGRESS_REWARD * 2  # Big jump
            elif progress > 0.05:
                worker_reward += self.config.PROGRESS_REWARD
            elif progress < -0.15:
                worker_reward += self.config.STAGNATION_PENALTY
            elif progress < -0.05:
                worker_reward += self.config.STAGNATION_PENALTY * 0.5
        
        if author_id:
            author_reward, _, author_reason = self.calculate_author_reward()
            if author_reward != 0:
                worker_reward += author_reward
                if self.current_step <= 5:
                    print(f"AUTH-R: +{author_reward:.1f} | {author_reason}")
        
        elif paper_id:
            citation_count = self.current_node.get('citationCount', 0)
            if citation_count >= 100:
                cit_bonus = min(np.log10(citation_count / 100.0) * 2.0, 5.0)
                worker_reward += cit_bonus
            
            year = self.current_node.get('year')
            if year and year >= 2020:
                recency_bonus = (year - 2020) * self.config.RECENCY_BONUS
                worker_reward += recency_bonus

            venue = self.current_node.get('venue', '') or self.current_node.get('publicationName', '')
            if venue:
                for top_venue in self.config.TOP_VENUES:
                    if top_venue.lower() in venue.lower():
                        worker_reward += 2.0
                        break
        
        if is_revisit:
            visit_count = self.node_visit_count.get(paper_id or author_id, 1)
            if visit_count > 2: 
                revisit_penalty = -5.0 * (visit_count - 2) 
                worker_reward += revisit_penalty
        
        if self.use_communities and (self.current_step % 5 == 0 or self.current_step <= 3):
            comm_reward, _, comm_reason = self.calculate_community_reward()
            if comm_reward != 0:
                worker_reward += comm_reward
                if self.current_step <= 5 or self.current_step % 5 == 0:
                    print(f"COMM-R: {comm_reward:+.1f} | {comm_reason}")
        
        if not is_revisit and semantic_sim < 0.6:  
            node_id = paper_id or author_id
            novelty_bonus = self.curiosity_module.get_novelty_bonus(node_id)
            current_state = await self._get_state()
            intrinsic = self.curiosity_module.compute_intrinsic_reward(
                current_state, node_emb, current_state
            )
            curiosity_total = min(novelty_bonus + intrinsic * 0.3, self.config.MAX_NOVELTY_BONUS)
            worker_reward += curiosity_total

        if self.query_intent and self.reward_mapper:
            h_reward, h_reason = self.reward_mapper.get_worker_reward(
                node=self.current_node,
                query_intent=self.query_intent,
                semantic_sim=semantic_sim
            )
            if h_reward != 0:
                worker_reward += h_reward
                if self.current_step <= 5:
                    print(f"DSPy-WKR: {h_reward:+.1f} | {h_reason}")
        
        worker_reward = np.clip(worker_reward, -50.0, 150.0) 
        
        self.previous_node_embedding = node_emb
        if semantic_sim > self.best_similarity_so_far:
            worker_reward += 20.0  
            self.best_similarity_so_far = semantic_sim
            self.episode_stats['max_similarity_achieved'] = semantic_sim
        
        done = self.current_step >= self.max_steps
        
        if done:
            diversity_reward = self._calculate_trajectory_rewards()
            worker_reward += diversity_reward
            print(f" TERM: Episode ended (step{self.current_step}/{self.max_steps})")
            print(f"REWARD: Trajectory diversity bonus = {diversity_reward:.1f}")
        
        self.trajectory_history.append({
            'node': self.current_node,
            'similarity': semantic_sim,
            'reward': worker_reward,
            'step': self.current_step,
            'community': self.current_community,
            'node_type': 'paper' if paper_id else 'author'
        })
        
        if self.current_step <= 3:
            title_str = str(self.current_node.get('title') or 'NA')[:50]
            name_str = str(self.current_node.get('name') or 'NA')[:50]
            display = title_str if paper_id else name_str
            print(f" DEBUG: Step {self.current_step} | sim={semantic_sim:.3f}, {'paper' if paper_id else 'author'}={display}")
        
        return await self._get_state(), worker_reward, done


    def _is_target_paper(self, current_title: str, target_title: str) -> bool:
        current = current_title.lower().strip()
        target = target_title.lower().strip()
        
        if current == target:
            return True
        
        if target in current or current in target:
            if len(target) > 3:
                return True
        
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, current, target).ratio()
        return similarity > 0.85


    async def _check_citation_relationship(
        self, 
        source_paper_id: str, 
        target_paper_id: str, 
        direction: str = 'cites'
    ) -> bool:
        """Check if source paper cites target paper (or vice versa)."""
        
        if hasattr(self, 'training_edge_cache'):
            edges = self.training_edge_cache.get(source_paper_id, [])
            for edge_type, dest_id in edges:
                if edge_type == direction and dest_id == target_paper_id:
                    return True
        
        # Fallback to Neo4j query 
        query = """
        MATCH (p1:Paper {paperId: $1})-[:CITES]->(p2:Paper {paperId: $2})
        RETURN count(*) > 0 as exists
        """
        
        try:
            result = await self.store._run_query_method(query, [source_paper_id, target_paper_id])
            if result and len(result) > 0:
                return result[0].get('exists', False)
        except:
            pass
        
        return False
    
    def get_episode_summary(self) -> Dict[str, Any]:
        summary = {
            **self.episode_stats,
            'path_length': len(self.trajectory_history),
            'relation_diversity': len(self.relation_types_used),
            'community_diversity_ratio': (
                self.episode_stats['unique_communities_visited'] / 
                max(1, self.episode_stats['total_nodes_explored'])
            ),
        }
        
        if hasattr(self, 'current_query_facets') and self.current_query_facets:
            summary['query_facets'] = self.current_query_facets
        elif hasattr(self, 'query_intent') and self.query_intent:
            summary['query_intent'] = {
                'target_entity': self.query_intent.target_entity if hasattr(self.query_intent, 'target_entity') else 'N/A',
                'operation': self.query_intent.operation if hasattr(self.query_intent, 'operation') else 'N/A',
            }
        else:
            summary['query_facets'] = None
        
        return summary


    async def get_valid_actions(self) -> List[Tuple[Dict, int]]:
        paper_id = self.current_node.get('paper_id')
        refs = await self.store.get_references_by_paper(paper_id)
        actions = [(self._normalize_node_keys(n), RelationType.CITES) for n in refs]
        cites = await self.store.get_citations_by_paper(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.CITED_BY) for n in cites]
        authors = await self.store.get_authors_by_paper_id(paper_id)
        actions += [(self._normalize_node_keys(n), RelationType.WROTE) for n in authors]
        return [(node, r_type) for node, r_type in actions if not self._is_visited(node)]

    async def step(self, action_node: Dict, action_relation: int):
        self.current_step += 1
        self.current_node = self._normalize_node_keys(action_node)
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        if paper_id:
            self.visited.add(paper_id)
        if author_id:
            self.visited.add(author_id)
        struct_reward = 1.0 if action_relation == self.current_intent else -0.5
        node_emb = await self._get_node_embedding(self.current_node)
        sem_reward = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
        total_reward = (0.6 * struct_reward) + (0.4 * sem_reward)
        done = self.current_step >= self.max_steps
        return await self._get_state(), total_reward, done
        
class MetaPathReward: 
    def _init__(self): 
        self.meta_paths = {
            'citation_chain': {
                'pattern': ['Paper', 'CITES', 'Paper', 'CITES', 'Paper'],
                'reward': 15.0,
                'description': 'Following citation chain'
            },
            'coauthor_bridge': {
                'pattern': ['Paper', 'WROTE', 'Author', 'WROTE', 'Paper'],
                'reward': 20.0,
                'description': 'Finding related work via shared author'
            },
            'venue_exploration': {
                'pattern': ['Paper', 'VENUE', 'Paper', 'VENUE', 'Paper'],
                'reward': 12.0,
                'description': 'Exploring same venue papers'
            },
            'influence_path': {
                'pattern': ['Paper', 'CITES', 'Paper', 'WROTE', 'Author'],
                'reward': 18.0,
                'description': 'Finding influential authors'
            },
            'temporal_forward': {
                'pattern': ['Paper(old)', 'CITED_BY', 'Paper(new)'],
                'reward': 10.0,
                'description': 'Following citations forward in time'
            }
        }
        
        self.current_path = []


    def add_step(self, node_type: str, relation_type: str):
        """Add a step to current path."""
        self.current_path.append((node_type, relation_type))
        
        # Keep only last 5 steps
        if len(self.current_path) > 5:
            self.current_path.pop(0)

    def check_meta_paths(self) -> Tuple[float , str]: 
        for path_name , path_info in self.meta_paths: 
            if self._matches_pattern(path_info['pattern']): 
                return path_info['reward'] , path_info['description']
        return 0.0 , ''
    
    def _matches_pattern(self , pattern: List[str]) -> bool: 
        if len(self.current_path) < len(pattern) // 2:
            return False 
        
        #TODO: Make more sophisticated
        recent_types = [node_type for node_type, _ in self.current_path[-(len(pattern)//2):]]
        pattern_types = [p for p in pattern if p in ['Paper', 'Author']]
        
        return recent_types == pattern_types