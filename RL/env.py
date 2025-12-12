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
        
        # Map old field names to new ones for compatibility
        if 'fields' in normalized and 'fieldsOfStudy' not in normalized:
            normalized['fieldsOfStudy'] = normalized['fields']
        if 'venue' not in normalized and 'publication_name' in normalized:
            normalized['venue'] = normalized['publication_name']
        
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
        
        # FIXED: Better starting paper selection
        if start_node_id:
            node = await self.store.get_paper_by_id(start_node_id)
            if not node:
                raise ValueError(f"Paper with ID {start_node_id} not found.")
            self.current_node = self._normalize_node_keys(node)
        else:
            # Try keyword matching first
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
    

    # In env.py, find get_node_embedding and update the abstract handling:

    async def _get_node_embedding(self, node: Dict[str, Any]) -> np.ndarray:
        """Get embedding for a node. ALWAYS returns valid ndarray, never None."""
        if not node:
            logging.warning("Empty node passed to function")
            return np.zeros(self.text_dim, dtype=np.float32)
        
        paper_id = node.get('paper_id')
        author_id = node.get('author_id')
        
        if paper_id and paper_id in self.precomputed_embeddings:
            return self.precomputed_embeddings[paper_id]
        if author_id and author_id in self.precomputed_embeddings: 
            return self.precomputed_embeddings[author_id]
        
        node_text = None 
        
        if paper_id: 
        
            title = str(node.get('title', '')) if node.get('title') else ''
            
            INVALID_TITLES = {'', 'NA', '...', 'Unknown', 'null', 'None', 'undefined'}
            
            if title and title not in INVALID_TITLES and len(title) >= 3:
                fields = str(node.get('fields', '')) if node.get('fieldsOfStudy') else ''
                keywords = ', '.join(fields) if isinstance(fields, list) else str(fields)
                abstract = str(node.get('abstract', '')) if node.get('abstract') else ''
                abstract = abstract or ''  # FIXED: Handle None
                pub_name = str(node.get('venue', '')) if node.get('venue') else ''
                
                parts = [title]
                if keywords:
                    parts.append(keywords)
                if abstract and len(abstract) > 20:
                    parts.append(abstract[:500])
                elif pub_name:
                    parts.append(pub_name)
                
                node_text = ' '.join(parts).strip()
            
                
        elif author_id: 
            name = str(node.get('name' , '')) if node.get('name') else ''
            
            INVALID_NAMES = {'' , 'NA' , '...' , 'Unknown' , 'NULL' , 'null' , 'None' , 'undefined' , 'XYZ'}
            if name and name not in INVALID_NAMES and len(name) > 1: 
                paper_count = node.get('paper_count', 0)
                h_index = node.get('h_index', 0)
                
                if paper_count or h_index:
                    node_text = f"{name}, {paper_count} papers, h-index {h_index}"
                else:
                    node_text = f"Author: {name}"

        if not node_text or len(node_text) < 5:
            if paper_id:
                node_text = f"Research paper {paper_id[-10:]}"
            else:
                if self.current_step <= 3:
                    logging.warning(f"Node has no usable text: {paper_id}")
                return np.zeros(self.text_dim, dtype=np.float32)
        
            
        try:
            if hasattr(self.encoder, 'encode_with_cache'):
                embedding = self.encoder.encode_with_cache(node_text, cache_keys=[paper_id])
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
            if refs:
                valid_relations.append(RelationType.CITES)
            
            cites = await self.store.get_citations_by_paper(paper_id)
            if cites:
                valid_relations.append(RelationType.CITED_BY)
            
            authors = await self.store.get_authors_by_paper_id(paper_id)
            if authors:
                valid_relations.append(RelationType.WROTE)
            
            keywords = self.current_node.get('fieldOfStudy' , '') or self.current_node.get('fields' , '')
            if keywords:
                valid_relations.append(RelationType.KEYWORD_JUMP)
            
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue:
                valid_relations.append(RelationType.VENUE_JUMP)
            
            older = await self.store.get_older_references(paper_id)
            if older:
                valid_relations.append(RelationType.OLDER_REF)
            
            newer = await self.store.get_newer_citations(paper_id)
            if newer:
                valid_relations.append(RelationType.NEWER_CITED_BY)
        
        elif author_id:
            papers = await self.store.get_papers_by_author_id(author_id)
            if papers:
                valid_relations.append(RelationType.AUTHORED)
            
            collabs = await self.store.get_collabs_by_author(author_id)
            if collabs:
                valid_relations.append(RelationType.COLLAB)
            
            second_collabs = await self.store.get_second_degree_collaborators(author_id , limit = 20)
            if second_collabs:
                valid_relations.append(RelationType.SECOND_COLLAB)
            
            influence = await self.store.get_influence_path_papers(author_id , limit = 10)
            if influence:
                valid_relations.append(RelationType.INFLUENCE_PATH)
        
        valid_relations.append(RelationType.STOP)
        
        return valid_relations
    
    async def manager_step(self, relation_type: int) -> Tuple[bool, float]:
        """Manager step with standard rewards."""
        self.pending_manager_action = relation_type
        
        relation = RelationType()
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
        
        # Query-aligned relation bonus
        if self.current_query_facets and self.current_query_facets.get('relation_focus'):
            focus = self.current_query_facets['relation_focus']
            
            relation_map = {
                RelationType.CITES: 'CITES',
                RelationType.CITED_BY: 'CITED_BY',
                RelationType.WROTE: 'WROTE',
                RelationType.AUTHORED: 'WROTE',
                RelationType.COLLAB: 'COLLAB',
                RelationType.SECOND_COLLAB: 'COLLAB',
            }
            
            current_relation_name = relation_map.get(relation_type)
            
            if current_relation_name == focus:
                manager_reward += 2.0
                logging.info(f"Manager chose query-aligned relation: {focus}")
            

            if self.current_query_facets.get('paper_operation') == 'citations':
                if relation_type == RelationType.CITED_BY:
                    manager_reward += 1.5
            elif self.current_query_facets.get('paper_operation') == 'references':
                if relation_type == RelationType.CITES:
                    manager_reward += 1.5
        
        if relation_type == self.current_intent:
            manager_reward += self.config.INTENT_MATCH_REWARD
        else:
            manager_reward += self.config.INTENT_MISMATCH_PENALTY
        
        if relation_type not in self.relation_types_used:
            manager_reward += self.config.DIVERSITY_BONUS
            self.relation_types_used.add(relation_type)
            self.episode_stats['unique_relation_types'] += 1
        
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        raw_nodes = []
        
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
            keywords = self.current_node.get('keywords', '')
            if keywords:
                keyword = keywords.split(',')[0].strip() if isinstance(keywords, str) else str(keywords[0])
                raw_nodes = await self.store.get_papers_by_keyword(keyword, limit=5, exclude_paper_id=paper_id)
        elif relation_type == RelationType.VENUE_JUMP and paper_id:
            venue = self.current_node.get('publication_name') or self.current_node.get('venue')
            if venue:
                raw_nodes = await self.store.get_papers_by_venue(venue, exclude_paper_id=paper_id)
        elif relation_type == RelationType.OLDER_REF and paper_id:
            raw_nodes = await self.store.get_older_references(paper_id)
        elif relation_type == RelationType.NEWER_CITED_BY and paper_id:
            raw_nodes = await self.store.get_newer_citations(paper_id)
        elif relation_type == RelationType.SECOND_COLLAB and author_id:
            raw_nodes = await self.store.get_second_degree_collaborators(author_id , limit=20)
        elif relation_type == RelationType.INFLUENCE_PATH and author_id:
            raw_nodes = await self.store.get_influence_path_papers(author_id , limit = 10)
        if relation_type == RelationType.STOP:
            if self.current_step < 5:
                return False, -5.0 

        print(f"  [MANAGER] Relation {relation_type}: fetched {len(raw_nodes)} raw nodes")
        
        normalized_nodes = [self._normalize_node_keys(node) for node in raw_nodes]
        
        INVALID_VALUES = {'', 'N/A', '...', 'Unknown', 'null', 'None', 'undefined'}
        valid_nodes = []
        
        for node in normalized_nodes:
            title = str(node.get('title', '')) if node.get('title') else ''
            name = str(node.get('name', '')) if node.get('name') else ''
            
            # Validate title (for papers)
            title_valid = (title and 
                        title not in INVALID_VALUES and 
                        len(title.strip()) > 3 and
                        not title.strip().startswith('N/A'))
            
            # Validate name (for authors)
            name_valid = (name and 
                        name not in INVALID_VALUES and 
                        len(name.strip()) > 2)
            
            # Keep node if either is valid
            if title_valid or name_valid:
                valid_nodes.append(node)
        
        self.available_worker_nodes = [(node, relation_type) for node in valid_nodes]
    
        
        self.available_worker_nodes = [
            (node, r_type) for node, r_type in self.available_worker_nodes 
            if not self._is_visited(node)
        ]
        
        print(f"  [MANAGER] After filtering: {len(self.available_worker_nodes)} valid unvisited nodes")
        num_available = len(self.available_worker_nodes)
        if num_available == 0:
            manager_reward += self.config.DEAD_END_PENALTY
            self.episode_stats['dead_ends_hit'] += 1
        elif num_available > 10:
            manager_reward += self.config.HIGH_DEGREE_BONUS
        elif num_available >= 5:
            manager_reward += 0.5
        
        return False, manager_reward


    async def manager_step_with_policy(self , state: np.ndarray) -> Tuple[bool , float , int]: 
        if not self.use_manager_policy or self.use_manager_policy is None:
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

        is_terminal , reward = self.manager_step(relation_type)

        self.manager_policy.update_policy(state, strategy, reward)
        
        return is_terminal, reward, relation_type        
        
    
    def _is_visited(self, node: Dict[str, Any]) -> bool:
        """Check if visited."""
        paper_id = node.get('paper_id')
        author_id = node.get('author_id')
        return (paper_id and paper_id in self.visited) or (author_id and author_id in self.visited)

    async def get_worker_actions(self) -> List[Tuple[Dict, int]]:
        """Get worker actions."""
        return self.available_worker_nodes
    

    async def worker_step(self, chosen_node: Dict) -> Tuple[np.ndarray, float, bool]:
        self.current_step += 1
        self.episode_stats['total_nodes_explored'] += 1
        
        self.current_node = self._normalize_node_keys(chosen_node)
        paper_id = self.current_node.get('paper_id')
        author_id = self.current_node.get('author_id')
        
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
        elif semantic_sim > 0.0: 
            worker_reward = semantic_sim * 10.0
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
            # Community switch bonus
            if self.previous_community and self.current_community != self.previous_community:
                worker_reward += self.config.COMMUNITY_SWITCH_BONUS
                
                # Extra bonus for discovering new community
                if self.current_community not in self.visited_communities:
                    worker_reward += 3.0
            
            # Penalty for staying stuck in same community
            if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
                penalty = self.config.COMMUNITY_STUCK_PENALTY
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                worker_reward += penalty
            
            # Penalty for looping back to previous communities
            if len(self.community_history) > 1:
                previous_communities = [c for c in self.community_history[:-1]]
                if self.current_community in previous_communities:
                    worker_reward += self.config.COMMUNITY_LOOP_PENALTY
            
            # Bonus for diverse community exploration
            unique_communities = len(self.visited_communities)
            if unique_communities >= 5:
                diversity_bonus = min((unique_communities - 4) * 2.0, 10.0)
                worker_reward += diversity_bonus
            
            # Community size bonus (medium-sized communities are best)
            if self.current_community:
                size = self.community_detector.get_community_size(self.current_community)
                if 5 <= size <= 50:
                    worker_reward += self.config.COMMUNITY_SIZE_BONUS
                elif 50 <= size <= 200:
                    worker_reward += 0.1
            
            # Temporal jump bonus (prefer newer papers)
            if len(self.community_history) > 1 and self.current_community:
                try:
                    prev_comm = self.community_history[-2]
                    prev_year = int(prev_comm.split('_')[0])
                    curr_year = int(self.current_community.split('_')[0])
                    
                    if prev_year < curr_year:
                        worker_reward += self.config.TEMPORAL_JUMP_BONUS
                    else:
                        worker_reward += self.config.TEMPORAL_JUMP_PENALTY
                except:
                    pass
        
        if author_id and self.use_communities and semantic_sim > 0.2:
            # Community switch bonus for authors
            if self.previous_community and self.current_community != self.previous_community:
                worker_reward += self.config.COMMUNITY_SWITCH_BONUS
                
                if self.current_community not in self.visited_communities:
                    worker_reward += 3.0
            
            # Stuck penalty for authors
            if self.steps_in_current_community >= self.config.STUCK_THRESHOLD:
                penalty = self.config.COMMUNITY_STUCK_PENALTY
                if self.steps_in_current_community >= self.config.SEVERE_STUCK_THRESHOLD:
                    penalty *= self.config.SEVERE_STUCK_MULTIPLIER
                worker_reward += penalty
            
            # Author-specific rewards
            # Prolific author bonus 
            paper_count = self.current_node.get('paperCount', 0) or self.current_node.get('paper_count', 0)
            if paper_count >= self.config.PROLIFIC_THRESHOLD:
                prolific_bonus = min(paper_count / 50.0, 5.0)  # Max +5
                worker_reward += prolific_bonus
            
            # H-index bonus 
            h_index = self.current_node.get('hIndex', 0) or self.current_node.get('h_index', 0)
            if h_index > 0:
                if author_id not in self.h_index_cache:
                    self.h_index_cache[author_id] = h_index
                    self.author_stats['max_h_index'] = max(self.author_stats['max_h_index'], h_index)
                
                h_index_bonus = min(h_index / 20.0, 3.0)  # Max +3
                worker_reward += h_index_bonus
            
            # Citation velocity bonus 
            citation_count = self.current_node.get('citationCount', 0)
            if citation_count > 1000:
                velocity_bonus = min(np.log10(citation_count) - 3, 2.0)  # Max +2
                worker_reward += velocity_bonus
            
            # Collaboration bonus 
            if self.steps_in_current_community == 1:  
                worker_reward += 2.0 
        
        # Node type switch bonus(Paper to Author transitions)
        if hasattr(self, 'previous_node_type'):
            previous_was_paper = self.previous_node_type == 'paper'
            current_is_paper = paper_id is not None
            
            if previous_was_paper != current_is_paper:
                worker_reward += self.config.NODE_TYPE_SWITCH 
        
        self.previous_node_type = 'paper' if paper_id else 'author'
        
        if not is_revisit and semantic_sim > 0.2:
            novelty_bonus = self.curiosity_module.get_novelty_bonus(paper_id or author_id)
            intrinsic = self.curiosity_module.compute_intrinsic_reward(
                await self._get_state(), node_emb, await self._get_state()
            )
            curiosity_total = min(novelty_bonus + intrinsic * 0.3, self.config.NOVELTY_BONUS)
            worker_reward += curiosity_total
        
        worker_reward -= 0.5 
        
        if semantic_sim > 0.65 and self.current_step <= 10:
            worker_reward += 30.0 

        worker_reward = np.clip(worker_reward, -50.0, 300.0)
        
        self.previous_node_embedding = node_emb
        
        done = self.current_step >= self.max_steps
        if done:
            print(f"  [TERM] Episode ended: step={self.current_step}/{self.max_steps}")
        
        next_state = await self._get_state()
        
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