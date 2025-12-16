from typing import Dict, Any, Optional , Tuple , Literal , List
from pydantic import BaseModel, Field
from enum import Enum , IntEnum
import logging
import numpy as np

class RelationType(IntEnum):
    """Copy from your env.py"""
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


class Constraint: 
    field: str  # 'venue', 'year', 'field', 'citation_count', etc.
    operator: Literal['equals', 'contains', 'greater_than', 'less_than', 'between', 'in_list']
    value: Any
    
    def __str__(self):
        return f"{self.field} {self.operator} {self.value}"

class QueryIntent: 
    target_entity = Literal['paper' , 'authors' , 'venues' , 'collaborations' , 'subjects' , 'communities']
    operations: Optional[Literal[
        'find',           # Find entities matching criteria
        'citations',      # Get citations
        'references',     # Get references
        'authors',        # Get authors of papers
        'papers',         # Get papers by authors
        'collaborators',  # Get collaborators
        'related',        # Get related entities
        'count',          # Count entities
        'traverse'        # Multi-hop traversal
    ]] = 'find'

    semantic: str = ""
    
    # Constraints (can be nested)
    constraints: List[Constraint] = Field(default_factory=list)
    
    # Multi-step chain (e.g., "authors who wrote papers that cite X")
    chain: Optional[List['QueryIntent']] = None
    
    # Original fields (for compatibility)
    author: Optional[str] = None
    paper_title: Optional[str] = None
    venue: Optional[str] = None
    temporal: Optional[List[int]] = None
    field_of_study: Optional[str] = None
    
    # Aggregation/filtering
    min_citation_count: Optional[int] = None
    max_citation_count: Optional[int] = None
    min_author_count: Optional[int] = None
    max_author_count: Optional[int] = None
    
    # Sorting
    sort_by: Optional[Literal['relevance', 'citations', 'year', 'h_index']] = None
    limit: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class ParserType(Enum):
    """Available parser implementations."""
    DSPY = "dspy"             
    LLM_MANUAL = "llm_manual"  
    RULE_BASED = "rule_based"  


class UnifiedQueryParser:
    """
    Unified interface for all query parsers with automatic fallback.
    """
    
    def __init__(
        self,
        primary_parser: ParserType = ParserType.DSPY,
        model: str = "gpt-3.5-turbo",
        fallback_on_error: bool = True
    ):
        """
        Args:
            primary_parser: Parser to use first
            model: LLM model name (for DSPy/LLM parsers)
            fallback_on_error: If True, fall back to simpler parser on failure
        """
        self.primary_parser_type = primary_parser
        self.fallback_on_error = fallback_on_error
        self.model = model
        
        # Initialize parsers lazily (only when needed)
        self._dspy_parser = None
        self._llm_parser = None
        self._rule_parser = None
        
        # Stats
        self.stats = {
            'dspy': {'success': 0, 'failure': 0},
            'llm_manual': {'success': 0, 'failure': 0},
            'rule_based': {'success': 0, 'failure': 0}
        }
        
        self._load_primary_parser()
    
    def _load_primary_parser(self):
        """Load the primary parser."""
        try:
            if self.primary_parser_type == ParserType.DSPY:
                self._dspy_parser = self._load_dspy()
            elif self.primary_parser_type == ParserType.LLM_MANUAL:
                self._llm_parser = self._load_llm_manual()
            else:  # RULE_BASED
                self._rule_parser = self._load_rule_based()
        except Exception as e:
            logging.warning(f"Failed to load {self.primary_parser_type.value}: {e}")
            if self.fallback_on_error:
                logging.info("Falling back to rule-based parser")
                self._rule_parser = self._load_rule_based()
    
    def _load_dspy(self):
        """Load DSPy optimized parser."""
        from .dspy_parser import OptimQueryParser
        
        logging.info("Loading DSPy optimized parser")
        return OptimQueryParser(
            model=self.model
        )
    
    def _load_llm_manual(self):
        """Load manual LLM parser."""
        from .queryparser import LLMQueryParser
        
        logging.info("✓ Loading manual LLM parser")
        return LLMQueryParser(model=self.model)
    
    def _load_rule_based(self):
        """Load rule-based parser."""
        from .queryparser import EnhancedQueryParser
        
        logging.info("✓ Loading rule-based parser")
        return EnhancedQueryParser()
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse query with automatic fallback.
        
        Returns:
            Facets dictionary with guaranteed structure
        """
        # Try primary parser
        if self.primary_parser_type == ParserType.DSPY and self._dspy_parser:
            try:
                facets = self._dspy_parser.parse(query)
                self.stats['dspy']['success'] += 1
                return self._validate_facets(facets, query)
            except Exception as e:
                logging.warning(f"DSPy parser failed: {e}")
                self.stats['dspy']['failure'] += 1
                if not self.fallback_on_error:
                    raise
        
        # Fallback to manual LLM
        if self._llm_parser is None:
            self._llm_parser = self._load_llm_manual()
        
        try:
            facets = self._llm_parser.parse_query(query)
            self.stats['llm_manual']['success'] += 1
            return self._validate_facets(facets, query)
        except Exception as e:
            logging.warning(f"LLM parser failed: {e}")
            self.stats['llm_manual']['failure'] += 1
            if not self.fallback_on_error:
                raise
        
        # Final fallback to rule-based (always works)
        if self._rule_parser is None:
            self._rule_parser = self._load_rule_based()
        
        facets = self._rule_parser.parse(query)
        self.stats['rule_based']['success'] += 1
        return self._validate_facets(facets, query)
    
    def _validate_facets(self, facets: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Ensure facets have all required fields with correct types.
        Adds 'original' field if missing.
        """
        required_fields = {
            'semantic': str,
            'temporal': (type(None), tuple, list),
            'author': (type(None), str),
            'author_search_mode': bool,
            'institutional': (type(None), str),
            'venue': (type(None), str),
            'intent': (type(None), str),
            'entities': list,
            'original': str,
            'paper_title': (type(None), str),
            'paper_search_mode': bool,
            'paper_operation': (type(None), str),
            'relation_focus': (type(None), str),
            'hop_depth': (type(None), int)
        }
        
        if 'original' not in facets:
            facets['original'] = original_query
        
        # Validate each field
        for field, expected_type in required_fields.items():
            if field not in facets:
                # Set default
                if field == 'original':
                    facets[field] = original_query
                elif field == 'entities':
                    facets[field] = []
                elif field == ['author_search_mode' , 'paper_search_mode']:
                    facets[field] = False
                elif field == 'hop_depth': 
                    facets[field] = None
                else:
                    facets[field] = None

            if field in ['author_search_mode', 'paper_search_mode']:
                if facets[field] is None:
                    facets[field] = False
                    logging.debug(f"Converting {field} from None to False")
        
        # Convert temporal to tuple if it's a list
        if facets['temporal'] and isinstance(facets['temporal'], list):
            facets['temporal'] = tuple(facets['temporal'])
        
        return facets
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser usage statistics."""
        total = sum(s['success'] + s['failure'] for s in self.stats.values())
        
        return {
            'total_queries': total,
            'by_parser': self.stats,
            'primary_parser': self.primary_parser_type.value
        }
    
    def print_stats(self):
        """Pretty print statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("QUERY PARSER STATISTICS")
        print("="*50)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Primary parser: {stats['primary_parser']}")
        print()
        
        for parser_name, counts in stats['by_parser'].items():
            total = counts['success'] + counts['failure']
            if total > 0:
                success_rate = 100 * counts['success'] / total
                print(f"{parser_name:12s}: {counts['success']:3d} success, {counts['failure']:3d} failure ({success_rate:.1f}%)")
        
        print("="*50 + "\n")
    

class QueryRewardCalculator:
    """Calculate rewards based on query facet matching."""
    
    def __init__(self, config):
        self.config = config
        self.relation_intent_map = {
            # Citation network exploration
            'citations': [RelationType.CITED_BY, RelationType.NEWER_CITED_BY],
            'references': [RelationType.CITES, RelationType.OLDER_REF],
            'citation_network': [RelationType.CITES, RelationType.CITED_BY],
            
            # Author exploration
            'author_works': [RelationType.WROTE, RelationType.AUTHORED],
            'collaborations': [RelationType.COLLAB, RelationType.SECOND_COLLAB],
            'author_influence': [RelationType.INFLUENCE_PATH],
            
            # Topic/field exploration
            'topic_exploration': [RelationType.KEYWORD_JUMP],
            'venue_exploration': [RelationType.VENUE_JUMP],
            
            # Temporal
            'recent_work': [RelationType.NEWER_CITED_BY],
            'foundational_work': [RelationType.OLDER_REF],
        }
    
        self.relation_purpose = {}
        for intent, relations in self.relation_intent_map.items():
            for rel in relations:
                if rel not in self.relation_purpose:
                    self.relation_purpose[rel] = []
                self.relation_purpose[rel].append(intent)

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Safely convert value to string, return empty string if None."""
        if value is None:
            return ''
        return str(value)
    
    def get_manager_reward(self , relation_type: int , query_facet: Dict[str , Any] , current_node: Dict[str , Any]) -> Tuple[float , str ]:
        reward = 0.0 
        reasons = []

        paper_op = query_facet.get('paper_operation')
        if paper_op: 
            if paper_op == 'citations' and relation_type == RelationType.CITED_BY: 
                reward += 20.0
                reasons.append("correct_operations:citations")
            elif paper_op == 'references' and relation_type == RelationType.CITES: 
                reward += 20.0
                reasons.append("correct_operation: references")
            elif paper_op == 'coauthors' and relation_type == RelationType.WROTE:
                reward += 20.0
                reasons.append("correct_operation:coauthors")
            elif paper_op == 'related' and relation_type in [RelationType.KEYWORD_JUMP, RelationType.VENUE_JUMP]:
                reward += 15.0
                reasons.append("correct_operation:related")
            elif paper_op in ['citations', 'references'] and relation_type not in [RelationType.CITES, RelationType.CITED_BY]:
                reward -= 10.0
                reasons.append("wrong_operation")
    
        relation_focus = query_facet.get('relation_focus')
        if relation_focus:
            focus_map = {
                'CITES': RelationType.CITES,
                'CITED_BY': RelationType.CITED_BY,
                'WROTE': RelationType.WROTE,
                'COLLAB': [RelationType.COLLAB, RelationType.SECOND_COLLAB],
                'PUBLISHED_IN': [RelationType.VENUE_JUMP],
                'SAME_COMMUNITY': [RelationType.KEYWORD_JUMP, RelationType.VENUE_JUMP],
            }


        target_relations = focus_map.get('relation_focus' , [])
        if not isinstance(target_relations , list):
            target_relations = [target_relations]

        if relation_type in target_relations:
            reward += 15.0
            reasons.append(f"relation_focus{relation_focus}")
        else:
            reward -= 8.0 
            reasons.append(f"wrong_focus{relation_focus}")


        intent = query_facet.get('intent')
        if intent and intent in self.relation_intent_map:
            target_relations = self.relation_intent_map[intent]
            if relation_type in target_relations:
                reward += 10.0
                reasons.append(f"intent_match:{intent}")


        if query_facet.get('author_search_mode'):
            if relation_type in [RelationType.WROTE, RelationType.AUTHORED, RelationType.COLLAB]:
                reward += 12.0
                reasons.append("author_search_mode")
            elif relation_type in [RelationType.CITES, RelationType.CITED_BY]:
                reward += 5.0 
                reasons.append("author_search_indirect")

            elif relation_type in [RelationType.COLLAB , RelationType.SECOND_COLLAB]:
                reward += 5.0 
                reasons.append("author search indirect")
                

        temporal = query_facet.get('temporal')
        if temporal:
            start_year, end_year = temporal
            current_year = 2024 
            
            if end_year >= current_year - 3:  
                if relation_type == RelationType.NEWER_CITED_BY:
                    reward += 8.0
                    reasons.append("temporal_recent")
                elif relation_type == RelationType.OLDER_REF:
                    reward -= 5.0
                    reasons.append("temporal_mismatch")
            
            elif end_year < 2010:
                if relation_type == RelationType.OLDER_REF:
                    reward += 8.0
                    reasons.append("temporal_classic")
                elif relation_type == RelationType.NEWER_CITED_BY:
                    reward -= 5.0
                    reasons.append("temporal_mismatch")



        venue = query_facet.get('venue')
        if venue and relation_type == RelationType.VENUE_JUMP:
            reward += 10.0
            reasons.append(f"venue_constraint:{venue}")


        hop_depth = query_facet.get('hop_depth')
        if hop_depth and hop_depth > 1:
            if relation_type in [RelationType.CITES, RelationType.CITED_BY]:
                reward += 5.0 * hop_depth
                reasons.append(f"multi_hop:{hop_depth}")
        
        reason_str = " | ".join(reasons) if reasons else "no_alignment"
        return reward, reason_str
    
    def get_worker_reward(self , node: Dict[str , Any] , query_facet: Dict[str , Any], semantic_sim: float) -> Tuple[Dict, str]: 
        reward = 0.0 
        reasons = []

        paper_id = node.get('paper_id') or node.get('paperId')
        author_id = node.get('author_id') or node.get('authorId')


        target_author = query_facet.get('author')
        if target_author and author_id:
            node_name = node.get('name', '').lower()
            if target_author.lower() in node_name or node_name in target_author.lower():
                reward += 80.0
                reasons.append("target_author_found")
        
        # Paper matching
        target_paper = query_facet.get('paper_title')
        if target_paper and paper_id:
            node_title = node.get('title', '').lower()
            if self._fuzzy_match(node_title, target_paper.lower()):
                reward += 80.0
                reasons.append("target_paper_found")


        temporal = query_facet.get('temporal')
        if temporal and paper_id:
            start_year, end_year = temporal
            paper_year = node.get('year')
            
            if paper_year:
                if start_year <= paper_year <= end_year:
                    reward += 15.0
                    reasons.append(f"temporal_match:{paper_year}")
                else:
                    reward -= 10.0
                    reasons.append(f"temporal_mismatch:{paper_year}")

        venue = query_facet.get('venue')
        if venue and paper_id:
            node_venue = node.get('venue', '') or node.get('publicationName', '')
            if venue.lower() in node_venue.lower():
                reward += 20.0
                reasons.append(f"venue_match:{venue}")


        min_citations = query_facet.get('citation_count_min')
        if min_citations and paper_id:
            citation_count = node.get('citationCount', 0)
            if citation_count >= min_citations:
                reward += 10.0
                reasons.append(f"citation_threshold:{citation_count}")
            else:
                reward -= 15.0
                reasons.append(f"low_citations:{citation_count}")
        
        # Author count constraint
        min_authors = query_facet.get('author_count_min')
        if min_authors and paper_id:
            # You'd need to fetch this from graph
            author_count = node.get('author_count', 0)
            if author_count >= min_authors:
                reward += 10.0
                reasons.append(f"author_threshold:{author_count}")


        entities = query_facet.get('entities', [])
        if entities and paper_id:
            node_fields = node.get('fieldsOfStudy', []) or node.get('fields', [])
            if isinstance(node_fields, list):
                for entity in entities:
                    for field in node_fields:
                        if entity.lower() in field.lower():
                            reward += 5.0
                            reasons.append(f"field_match:{entity}")
                            break
        
        author_operation = query_facet.get('author_operation')
        if author_operation == 'collaborations' and author_id:
            # Check if this is a collaborator (not the target author)
            if target_author and node.get('name', '').lower() != target_author.lower():
                reward += 20.0
                reasons.append("collaborator_found")
        
        reason_str = " | ".join(reasons) if reasons else "semantic_only"
        return reward, reason_str
    
    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
    
        if str1 == str2:
            return True
        
        if len(str2) > 3 and (str2 in str1 or str1 in str2):
            return True
        
        # Levenshtein similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, str1, str2).ratio()
        return similarity >= threshold
    
    