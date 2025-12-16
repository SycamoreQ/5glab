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
    INTENT_MATCH_REWARD = 2.0
    INTENT_MISMATCH_PENALTY = -0.5
    DIVERSITY_BONUS = 0.5
    
    SEMANTIC_WEIGHT = 5.0
    NOVELTY_BONUS = 5.0
    DEAD_END_PENALTY = -0.5
    REVISIT_PENALTY = -0.3
    HIGH_DEGREE_BONUS = 1.0
    CITATION_COUNT_BONUS = 0.1
    RECENCY_BONUS = 0.2
    PROGRESS_REWARD = 2.0
    STAGNATION_PENALTY = -3.0
    GOAL_REACHED_BONUS = 5.0
    
    COMMUNITY_SWITCH_BONUS = 5.0       # Bonus for jumping to different community
    COMMUNITY_STUCK_PENALTY = -2.5    # Penalty per step stuck in same community
    COMMUNITY_LOOP_PENALTY = -1.0      # Severe penalty for returning to previous community
    DIVERSE_COMMUNITY_BONUS = 3.0    # Bonus for visiting many unique communities
    TEMPORAL_JUMP_BONUS = 0.7        # Bonus for moving to a community which is recent 
    TEMPORAL_JUMP_PENALTY = -0.1        # Bonus for doing the opposite 
    COMMUNITY_SIZE_BONUS = 0.3         # Bonus for not being in small communities
    NODE_TYPE_SWITCH = 0.1

    PROLIFIC_AUTHOR_BONUS = 0.4         # Bonus for authors with many papers
    PROLIFIC_THRESHOLD = 20             # Min papers for "prolific"
    COLLABORATION_BONUS = 0.3           # Bonus for well-connected authors
    COLLABORATION_THRESHOLD = 10        # Min collaborators
    AUTHOR_H_INDEX_BONUS = 0.35         # Bonus for high h-index (if available)
    INSTITUTION_QUALITY_BONUS = 0.25    # Bonus for top institutions 
    CITATION_VELOCITY_BONUS = 0.3
    
    STUCK_THRESHOLD = 2               # Steps in same community = "stuck"
    SEVERE_STUCK_THRESHOLD = 4         # Very stuck threshold
    SEVERE_STUCK_MULTIPLIER = 2.0      # Multiply penalty when severely stuck
    MAX_NOVELTY_BONUS = 10.0 


    TOP_VENUES = {
    'Nature', 'Science', 'Cell', 'PNAS',
    'CVPR', 'ICCV', 'ECCV', 'NeurIPS', 'ICML', 'ICLR',
    'ACL', 'EMNLP', 'NAACL', 'TACL',
    'SIGIR', 'WWW', 'KDD', 'ICDE',
    'AAAI', 'IJCAI', 'AAMAS'
    }

    TOP_INSTITUTIONS = {
    'MIT', 'Stanford', 'Berkeley', 'CMU', 'Harvard',
    'Oxford', 'Cambridge', 'ETH', 'Imperial',
    'Google', 'Microsoft', 'Meta', 'DeepMind', 'OpenAI'
    }


class AdvancedGraphTraversalEnv:
    """
    Community-aware hierarchical RL environment.
    Tracks and penalizes agent for getting stuck in local graph communities.
    """
    
    def __init__(self, store, embedding_model_name="all-MiniLM-L6-v2", 
                use_communities=True , use_feedback = True , use_llm_parser = True , parser_type: str = 'dspy' , 
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
        
        if self.use_communities and self.use_authors:
            self.current_comm_node = None
            self.prev_comm_node = None
            self.community_detector = CommunityDetector(store)
            if not self.community_detector.load_cache():
                print("No community cache found. Run: python -m RL.community_detection")
                print("Continuing without community rewards...")
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
        }.get(parser_type, ParserType.DSPY)
        
        self.query_parser = UnifiedQueryParser(
            primary_parser=parser_type_enum,
            model="llama3.2",
            fallback_on_error=True 
        )
        
        self.query_reward_calc = QueryRewardCalculator(self.config)
        self.current_query_facets = None

        
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

        self.curiosity_module = CuriosityModule()

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


    def _calculate_community_reward(self) -> Tuple[Tuple[float , str] , str]:
        """
        Calculate reward/penalty based on community exploration patterns.
        Returns: (node_type_reward, node_type) , reason
        """
        if not self.use_communities or self.current_community is None:
            return (0.0, "None") , "no_community"
        
        if self.current_community is None:
            if self.previous_community is not None:
                return (-0.1, "None") ,  "left_cache"
            return (0.0, "None") ,  "no_community"
        
        if len(self.community_history) < 2:
            return (0.0 , "Papers") ,  "no temporal history recorded for papers"
        
        if len(self.author_hist) < 2: 
            return (0.0 , "Authors") , "no temporal history recorded for authors"
    
        paper_reward = 0.0
        author_reward = 0.0
        reasons = []

        paper_id = self.current_comm_node.get('paper_id')
        author_id = self.current_comm_node.get('author_id')

        if self.steps_in_current_community == 1 and self.previous_community: 
            if self.previous_community != self.current_community: 
                if paper_id : 
                    paper_reward += self.config.COMMUNITY_SWITCH_BONUS
                
                else: 
                    author_reward += self.config.COMMUNITY_SWITCH_BONUS


        if self.steps_in_current_community == 1 and self.previous_community is not None:
            if self.current_community != self.previous_community:
                if paper_id:
                    paper_reward += self.config.COMMUNITY_SWITCH_BONUS
                    reasons.append(f"switch_bonus:+{self.config.COMMUNITY_SWITCH_BONUS:.2f}")

                elif author_id: 
                    author_reward += self.config.COMMUNITY_SWITCH_BONUS
                    reasons.append(f"switch_bonus:+{self.config.COMMUNITY_SWITCH_BONUS}")
                    
    
        if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
            paper_penalty = self.config.COMMUNITY_STUCK_PENALTY
            author_penalty = self.config.COMMUNITY_STUCK_PENALTY

            if paper_id: 
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    paper_penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                    reasons.append(f"severe_stuck:{paper_penalty:.2f}")
                else: 
                    reasons.append(f"stuck: {paper_penalty}")

            elif author_id: 
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    paper_penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                    reasons.append(f"severe_stuck:{paper_penalty:.2f}")        
                else:
                    reasons.append(f"stuck:{author_penalty:.2f}")
            
            paper_reward += paper_penalty
            author_reward += author_penalty
        
        if len(self.community_history) and len(self.author_hist)> 1:
            if paper_id:
                previous_communities = [c for _, c in self.community_history[:-1]]
                if self.current_community in previous_communities:
                    paper_reward += self.config.COMMUNITY_LOOP_PENALTY
                    reasons.append(f"loop:{self.config.COMMUNITY_LOOP_PENALTY:.2f}")

            if author_id:
                previous_communities = [c for _, c in self.community_history[:-1]]
                if self.current_community in previous_communities:
                    paper_reward += self.config.COMMUNITY_LOOP_PENALTY
                    reasons.append(f"loop:{self.config.COMMUNITY_LOOP_PENALTY:.2f}")
                
            if self.current_community and self.current_author: 
                visit_count_paper = self.community_visit_count[self.current_community]

            if self.current_author: 
                visit_count_author = self.author_visit_count[self.current_author]
                
                if visit_count_paper and visit_count_author > 2: 
                    if paper_id:
                        paper_reward += self.config.REVISIT_PENALTY*(visit_count_paper - 2)
                        reasons.append(f"revisit penalty{self.config.REVISIT_PENALTY:.2f}")
                    if author_id:
                        author_reward += self.config.REVISIT_PENALTY*(visit_count_author -2)
                        reasons.append(f"revisit penalty{self.config.REVISIT_PENALTY}")
                    
        unique_paper_communities = len(self.community_visit_count)
        unique_author_communities = len(self.author_visit_count)
        if unique_paper_communities and unique_author_communities >= 3:
            if paper_id: 
                diversity_bonus = self.config.DIVERSE_COMMUNITY_BONUS * (unique_paper_communities - 2)
                paper_reward += diversity_bonus
                reasons.append(f"diversity:+{diversity_bonus:.2f}")
            if author_id: 
                diversity_bonus = self.config.DIVERSE_COMMUNITY_BONUS*(unique_paper_communities-2)
                author_reward += diversity_bonus
                reasons.append(f"diversity+{self.config.DIVERSE_COMMUNITY_BONUS}")

        if paper_id:
            size = self.community_detector.get_community_size(self.current_community)
        
            if 5 <= size <= 50: 
                paper_reward += self.config.COMMUNITY_SIZE_BONUS
                reasons.append(f"size:{self.config.COMMUNITY_SIZE_BONUS:.2f}")

            elif 50 <= size <= 200: 
                paper_reward += 0.1 

        elif author_id:
            size = self.community_detector.get_community_size(self.current_community)
        
            if 5 <= size <= 50: 
                author_reward += self.config.COMMUNITY_SIZE_BONUS
                reasons.append(f"size:{self.config.COMMUNITY_SIZE_BONUS:.2f}")

            elif 50 <= size <= 200: 
                author_reward += 0.1 
        
        if paper_id: 

            prev_comm = self.community_history[-2][1]
            
            try: 
                prev_year = int(prev_comm.split('_')[0])
                curr_year = int(self.current_community.split('_')[0])

                if prev_year < curr_year: 
                    paper_reward += self.config.TEMPORAL_JUMP_BONUS
                else: 
                    paper_reward += self.config.TEMPORAL_JUMP_PENALTY
            
            except: 
                pass 

        elif author_id:
            prev_comm = self.community_history[-2][1]
            
            try: 
                prev_year = int(prev_comm.split('_')[0])
                curr_year = int(self.current_community.split('_')[0])

                if prev_year < curr_year: 
                    paper_reward += self.config.TEMPORAL_JUMP_BONUS
                else: 
                    paper_reward += self.config.TEMPORAL_JUMP_PENALTY
            
            except: 
                pass 

        if self.use_communities:
            current_comm = self.community_detector.get_community(self.current_paper_id)
            
            new_communities_available = 0
            for node_dict, _ in self.available_worker_nodes:
                node_id = node_dict.get('paper_id')
                if node_id:
                    node_comm = self.community_detector.get_community(node_id)
                    if node_comm and node_comm != current_comm:
                        if node_comm not in self.visited:
                            new_communities_available += 1
            
            if new_communities_available > 0:
                if paper_id:
                    paper_reward += 1.0 * min(new_communities_available, 3)  
                else:
                    author_reward += 1.0 * min(new_communities_available, 3)  
                
            elif current_comm and len(self.visited) < 5:
                if paper_id:
                    paper_reward += 0.3 
                else: 
                    author_reward += 0.3

        reason_str = ", ".join(reasons) if reasons else "none"
        return ((paper_reward , "Paper") , reason_str) if paper_id else ((author_reward , "Author") , reason_str)
    

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
         
        

    async def reset(self, query: str, intent: int, start_node_id: str = None):
        self.query_embedding = self.encoder.encode(query)
        self.current_intent = intent
        self.visited = set()
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
        
        self.current_query_facets = self.query_parser.parse(query)
        
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
                # Fallback: field-of-study query
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
        return np.concatenate([self.query_embedding, node_emb, intent_vec])


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
        """Manager step with IMPROVED action sampling using HybridAttentionSelector."""
        self.pending_manager_action = relation_type
        
        if relation_type == RelationType.STOP:
            self.current_step = self.max_steps
            if self.previous_node_embedding is None:
                return True, -1.0
            
            current_sim = np.dot(self.query_embedding, self.previous_node_embedding) / \
                         (np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9)
            
            if current_sim >= 0.7:
                return True, self.config.GOAL_REACHED_BONUS
            elif current_sim >= 0.4:
                return True, 0.5
            else:
                return True, -1.0
        
        manager_reward = 0.0
        
        paper_id = self.current_node.get('paper_id') or self.current_node.get('paperId')
        author_id = self.current_node.get('author_id') or self.current_node.get('authorId')
        raw_nodes = []

        if hasattr(self, 'training_edge_cache') and self.training_edge_cache and paper_id:
            edges = self.training_edge_cache.get(paper_id, [])
            if relation_type in (RelationType.CITES, RelationType.CITED_BY):
                want = 'cites' if relation_type == RelationType.CITES else 'cited_by'
                target_ids = [tid for et, tid in edges if et == want]
                raw_nodes = []
                for tid in target_ids:
                    n = await self.store.get_paper_by_id(tid)
                    if n:
                        raw_nodes.append(n)

        elif relation_type == RelationType.WROTE and paper_id:
            raw_nodes = await self.store.get_authors_by_paperid(paper_id)
        elif relation_type == RelationType.AUTHORED and author_id:
            raw_nodes = await self.store.get_papers_by_authorid(author_id)
        elif relation_type == RelationType.COLLAB and author_id:
            raw_nodes = await self.store.get_collabs_by_author(author_id)
        elif relation_type == RelationType.KEYWORD_JUMP and paper_id:
            keywords = self.current_node.get('fieldsOfStudy') or self.current_node.get('fields') or self.current_node.get('keywords') or []
            if keywords:
                keyword = keywords[0]
                raw_nodes = await self.store.get_papers_by_keyword(keyword, limit=20, exclude_paperid=paper_id)
        elif relation_type == RelationType.VENUE_JUMP and paper_id:
            venue = self.current_node.get('publicationName') or self.current_node.get('venue')
            if venue:
                raw_nodes = await self.store.get_papers_by_venue(venue, exclude_paperid=paper_id)
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
        
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]

        
        valid_unvisited = []
        valid_embeddings = []

        for node in normalized_nodes:
            pid = node.get('paper_id') or node.get('paperId')
            if self.require_precomputed_embeddings and pid and pid not in self.precomputed_embeddings:
                continue

            if self._is_visited(node):
                continue

            emb = await self._get_node_embedding(node)
            if emb is None or not isinstance(emb, np.ndarray):
                continue
            if np.allclose(emb, 0.0):
                continue

            valid_unvisited.append(node)
            valid_embeddings.append(emb)

        
        print(f"  [MANAGER] After filtering: {len(valid_unvisited)} valid unvisited nodes")
        
        MAX_WORKER_ACTIONS = 20  
        
        if len(valid_unvisited) > MAX_WORKER_ACTIONS:
            if self.use_attention and hasattr(self, 'attention_selector'):
                ranked_actions = self.attention_selector.get_ranked_actions(
                    query_emb=self.query_embedding,
                    candidate_nodes=valid_unvisited,
                    candidate_embs=valid_embeddings,
                    relation_types=[relation_type] * len(valid_unvisited),
                    current_state=await self._get_state() if hasattr(self, 'get_state') else None,
                    current_community=self.current_community if self.use_communities else None,
                    community_history=self.community_history if self.use_communities else None,
                    community_sizes={comm: self.community_detector.get_community_size(comm) 
                                   for comm in set(self.community_history)} if self.use_communities and self.community_detector else None,
                    community_detector=self.community_detector if self.use_communities else None,
                    top_k=MAX_WORKER_ACTIONS
                )
                
                valid_unvisited = [action['node'] for action in ranked_actions]
                print(f"  [MANAGER] Attention-based sampling: selected {len(valid_unvisited)} actions")
            else:
                sims = []
                for emb in valid_embeddings:
                    sim = np.dot(self.query_embedding, emb) / \
                          (np.linalg.norm(self.query_embedding) * np.linalg.norm(emb) + 1e-9)
                    sims.append(sim)
                
                top_k = int(MAX_WORKER_ACTIONS * 0.6)
                top_indices = np.argsort(sims)[-top_k:]
                
                # Random 40% for exploration
                remaining = [i for i in range(len(valid_unvisited)) if i not in top_indices]
                random_k = MAX_WORKER_ACTIONS - top_k
                if remaining and random_k > 0:
                    random_indices = np.random.choice(remaining, size=min(random_k, len(remaining)), replace=False)
                    selected_indices = np.concatenate([top_indices, random_indices])
                else:
                    selected_indices = top_indices
                
                valid_unvisited = [valid_unvisited[i] for i in selected_indices]
                print(f"  [MANAGER] Similarity-based sampling: selected {len(valid_unvisited)} actions")
        
        self.available_worker_nodes = [(node, relation_type) for node in valid_unvisited]
        
        num_available = len(self.available_worker_nodes)
        if num_available == 0:
            manager_reward = self.config.DEAD_END_PENALTY
            self.episode_stats['dead_ends_hit'] += 1
            self.current_step = self.max_steps
            return True, manager_reward
        
        elif num_available >= 10:
            manager_reward = self.config.HIGH_DEGREE_BONUS
        elif num_available >= 5:
            manager_reward = 0.5
        
        return False, manager_reward



    async def manager_step_with_policy(self , state: np.ndarray) -> Tuple[bool , float , int]: 
        if not hasattr(self, 'manager_policy') or self.manager_policy is None:
            relation_type = 1 if 1 in await self.get_manager_actions() else None
            if relation_type is None: 
                return True, -1.0, None 
            is_terminal, reward = await self.manager_step(relation_type)


        manager_actions = await self.get_manager_actions()
        if not manager_actions:
            return True, -1.0, None
        
        episode_progress = self.current_step/self.max_steps

        strategy = self.manager_policy.select_strategy(
            state, 
            episode_progress,
            visited_communities= len(self.visited_communities),
            current_reward= sum(t['reward'] for t in self.trajectory_history) 
        )

        env_state = {
            'current_community': self.current_community,
            'visited_communities': self.visited_communities
        }
        
        relation_type = self.manager_policy.get_relation_for_strategy(
            strategy,
            manager_actions,
            env_state
        )

        is_terminal , reward = await self.manager_step(relation_type)

        self.manager_policy.update_policy(state, strategy, reward)
        
        return is_terminal, reward, relation_type        
        
    
    def _is_visited(self, node: Dict[str, Any]) -> bool:
        """Check if visited."""
        paper_id = node.get('paper_id')
        author_id = node.get('author_id')
        return (paper_id and paper_id in self.visited) or (author_id and author_id in self.visited)
    

    async def get_worker_actions(self):
        if not self.current_node:
            return []
        
        paper_id = self.current_node.get('paperId') or self.current_node.get('paper_id')
        if not paper_id:
            return []
        
        actions = []
        
        if hasattr(self, 'training_edge_cache') and self.training_edge_cache:
            edges = self.training_edge_cache.get(paper_id, [])
            
            print(f"  [WORKER] Found {len(edges)} edges in cache for {paper_id[:10]}...")
            
            for edge_type, target_id in edges:
                if target_id in self.visited:
                    continue
                
                try:
                    target_paper = await self.store.get_paper_by_id(target_id)
                    if target_paper:
                        normalized = self._normalize_node_keys(target_paper)
                        pid = normalized.get('paper_id') or normalized.get('paperId')
                        if self.require_precomputed_embeddings and pid and pid not in self.precomputed_embeddings:
                            continue
                        emb = await self._get_node_embedding(normalized)
                        if emb is None or not isinstance(emb, np.ndarray) or np.allclose(emb, 0.0):
                            continue
                        
                        if edge_type == 'cites':
                            actions.append((normalized, RelationType.CITES))
                        elif edge_type == 'cited_by':
                            actions.append((normalized, RelationType.CITED_BY))
                except Exception as e:
                    print(f"  [WORKER] Error fetching paper {target_id}: {e}")
                    continue
            
            print(f"  [WORKER] Returning {len(actions)} valid actions from cache")
            return actions
        
        else:
            print(f"  [WORKER] No edge cache, querying Neo4j...")
            
            try:
                refs = await self.store.get_references_by_paper(paper_id)
                actions = [(self._normalize_node_keys(n), RelationType.CITES) for n in refs]
            except:
                refs = []

            try:
                cites = await self.store.get_citations_by_paper(paper_id)
                actions += [(self._normalize_node_keys(n), RelationType.CITED_BY) for n in cites]
            except:
                pass
            
            actions = [(node, r_type) for node, r_type in actions if not self._is_visited(node)]
            
            print(f"  [WORKER] Returning {len(actions)} valid actions from Neo4j")
            return actions
        

    async def worker_step(self, chosen_node: Dict) -> Tuple[np.ndarray, float, bool]:
        """Enhanced worker step with balanced rewards."""
        self.current_step += 1
        self.episode_stats['total_nodes_explored'] += 1
        
        self.current_node = self._normalize_node_keys(chosen_node)
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
        # Track revisits
        is_revisit = False
        if paper_id and paper_id in self.visited:
            is_revisit = True
            self.episode_stats['revisits'] += 1
        if author_id and author_id in self.visited:
            is_revisit = True
            self.episode_stats['revisits'] += 1
        
        if paper_id:
            self.visited.add(paper_id)
            self._update_community_tracking(paper_id)
        if author_id:
            self.visited.add(author_id)
            self._update_community_tracking(author_id)
        
        node_emb = await self._get_node_embedding(self.current_node)
        if node_emb is None or not isinstance(node_emb, np.ndarray):
            node_emb = np.zeros(self.text_dim, dtype=np.float32)
        
        semantic_sim = np.dot(self.query_embedding, node_emb) / (
            np.linalg.norm(self.query_embedding) * np.linalg.norm(node_emb) + 1e-9
        )
        
        if self.current_step <= 3:
            title = self.current_node.get('title', 'NA')[:50]
            name = self.current_node.get('name', 'NA')[:50]
            display = title if paper_id else name
            print(f"  [DEBUG] Step {self.current_step}: sim={semantic_sim:.3f}, {'paper' if paper_id else 'author'}={display}")

        if semantic_sim > self.best_similarity_so_far:
            self.best_similarity_so_far = semantic_sim
            self.episode_stats['max_similarity_achieved'] = semantic_sim

        if semantic_sim > 0.7:
            worker_reward = 200.0  # Jackpot
        elif semantic_sim > 0.5:
            worker_reward = 50.0 + 150.0 * ((semantic_sim - 0.5) / 0.2) ** 2
        elif semantic_sim > 0.3:
            worker_reward = 10.0 + 40.0 * ((semantic_sim - 0.3) / 0.2) ** 2
        elif semantic_sim > 0.15:
            worker_reward = (semantic_sim - 0.15) * 20.0
        else:
            worker_reward = -20.0 + semantic_sim * 50.0
        
        if self.previous_node_embedding is not None:
            prev_sim = np.dot(self.query_embedding, self.previous_node_embedding) / (
                np.linalg.norm(self.query_embedding) * np.linalg.norm(self.previous_node_embedding) + 1e-9
            )
            
            progress = semantic_sim - prev_sim
            if progress > 0.15:
                worker_reward += 40.0
            elif progress > 0.05:
                worker_reward += 20.0
            elif progress > 0:
                worker_reward += 10.0
            elif progress < -0.1:
                worker_reward -= 30.0
            elif progress < -0.05:
                worker_reward -= 15.0

        if is_revisit:
            worker_reward -= 50.0
        
        if paper_id and self.use_communities and semantic_sim > 0.3:
            # Community switch bonus (+5)
            if self.previous_community and self.current_community != self.previous_community:
                worker_reward += self.config.COMMUNITY_SWITCH_BONUS
                
                # New community discovery (+3)
                if self.current_community not in self.visited_communities:
                    worker_reward += 3.0
            
            # Stuck penalty (escalating)
            if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
                penalty = self.config.COMMUNITY_STUCK_PENALTY
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                worker_reward += penalty  # Negative value
            
            # Loop penalty (-5)
            if len(self.community_history) > 1:
                previous_communities = [c for c in self.community_history[:-1]]
                if self.current_community in previous_communities:
                    worker_reward += self.config.COMMUNITY_LOOP_PENALTY
            
            # Diversity bonus (Max: +10)
            unique_communities = len(self.visited_communities)
            if unique_communities >= 5:
                diversity_bonus = min((unique_communities - 4) * 2.0, 10.0)
                worker_reward += diversity_bonus
            
            # Community size bonus (+2)
            if self.current_community:
                size = self.community_detector.get_community_size(self.current_community)
                if 5 <= size <= 50:  # Sweet spot
                    worker_reward += self.config.COMMUNITY_SIZE_BONUS
                elif 50 <= size <= 200:
                    worker_reward += 0.5
            
            # Temporal jump bonus (+3 for newer, -2 for older)
            if len(self.community_history) > 1 and self.current_community:
                try:
                    prev_comm = self.community_history[-2]
                    curr_comm_parts = self.current_community.split('_')
                    prev_comm_parts = prev_comm.split('_')
                    
                    if curr_comm_parts[0] == 'L': 
                        curr_year = int(curr_comm_parts[2])
                        prev_year = int(prev_comm_parts[2])
                    else: 
                        curr_year = int(curr_comm_parts[1])
                        prev_year = int(prev_comm_parts[1])
                    
                    if curr_year > prev_year:
                        worker_reward += self.config.TEMPORAL_JUMP_BONUS
                    elif curr_year < prev_year:
                        worker_reward += self.config.TEMPORAL_JUMP_PENALTY
                except:
                    pass
        
        if author_id and self.use_communities and semantic_sim > 0.2:
            # Community switch for authors (+5)
            if self.previous_community and self.current_community != self.previous_community:
                worker_reward += self.config.COMMUNITY_SWITCH_BONUS
                
                if self.current_community not in self.visited_communities:
                    worker_reward += 3.0
            
            # Stuck penalty
            if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
                penalty = self.config.COMMUNITY_STUCK_PENALTY
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                worker_reward += penalty
            
            # Prolific author bonus (Max: +5)
            paper_count = self.current_node.get('paperCount', 0) or self.current_node.get('paper_count', 0)
            if paper_count >= self.config.PROLIFIC_THRESHOLD:
                prolific_bonus = min(paper_count / 50.0, 5.0)
                worker_reward += prolific_bonus
            
            # H-index bonus (Max: +3)
            h_index = self.current_node.get('hIndex', 0) or self.current_node.get('h_index', 0)
            if h_index > 0:
                if author_id not in self.h_index_cache:
                    self.h_index_cache[author_id] = h_index
                    self.author_stats['max_h_index'] = max(self.author_stats['max_h_index'], h_index)
                
                h_index_bonus = min(h_index / 20.0, 3.0)
                worker_reward += h_index_bonus
            
            # Citation velocity bonus (Max: +2)
            citation_count = self.current_node.get('citationCount', 0)
            if citation_count > 1000:
                velocity_bonus = min(np.log10(citation_count) - 3, 2.0)
                worker_reward += velocity_bonus
            
            # Collaboration bonus (+2)
            if self.steps_in_current_community == 1:
                worker_reward += 2.0

        if hasattr(self, 'previous_node_type'):
            previous_was_paper = self.previous_node_type == 'paper'
            current_is_paper = paper_id is not None
            
            if previous_was_paper != current_is_paper:
                worker_reward += self.config.NODE_TYPE_SWITCH
        
        self.previous_node_type = 'paper' if paper_id else 'author'
        

        if not is_revisit and semantic_sim > 0.2:
            node_id = paper_id or author_id
            novelty_bonus = self.curiosity_module.get_novelty_bonus(node_id)
            
            # Intrinsic curiosity (forward model prediction error)
            current_state = await self._get_state()
            intrinsic = self.curiosity_module.compute_intrinsic_reward(
                current_state, node_emb, current_state
            )
            
            curiosity_total = min(novelty_bonus + intrinsic * 0.3, self.config.MAX_NOVELTY_BONUS)
            worker_reward += curiosity_total
        
        worker_reward -= 0.5
        
        if semantic_sim > 0.65 and self.current_step <= 10:
            worker_reward += 30.0
        if self.use_communities:
            community_reward = self._calculate_community_reward()
            worker_reward += community_reward


        worker_reward = np.clip(worker_reward, -100.0, 250.0)
        
        # Update previous state
        self.previous_node_embedding = node_emb
        
        # Check if episode done
        done = self.current_step >= self.max_steps
        if done:
            # ADD TRAJECTORY DIVERSITY BONUS AT END
            diversity_reward = self._calculate_trajectory_rewards()
            worker_reward += diversity_reward
            
            print(f"  [TERM] Episode ended: step={self.current_step}/{self.max_steps}")
            print(f"  [REWARD] Trajectory diversity bonus: +{diversity_reward:.1f}")
        
        # Get next state
        next_state = await self._get_state()
        
        # Log trajectory
        self.trajectory_history.append({
            'node': self.current_node,
            'similarity': semantic_sim,
            'reward': worker_reward,
            'step': self.current_step,
            'community': self.current_community,
            'node_type': 'paper' if paper_id else 'author'
        })
        
        return next_state, worker_reward, done





    
    def get_diverse_results(self , trajectory: List[Dict]) -> List[Dict]: 
        if not self.use_feedback: 
            return trajectory
        
        papers_with_emb = []
        for step in trajectory:
            node = step['node']
            if node.get('paper_id'):
                papers_with_emb.append({
                    **node,
                    'embedding': step.get('node_embedding')  
                })
        
        diverse_papers = self.diversity_selector.select_diverse_papers(
            papers_with_emb,
            self.query_embedding,
            k=10
        )
        
        return diverse_papers
    

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get detailed episode summary with community stats."""
        return {
            **self.episode_stats,
            'path_length': len(self.trajectory_history),
            'relation_diversity': len(self.relation_types_used),
            'community_diversity_ratio': (
                self.episode_stats['unique_communities_visited'] / 
                max(1, self.episode_stats['total_nodes_explored'])
            ),
            'query_facets': self.current_query_facets 
        }

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