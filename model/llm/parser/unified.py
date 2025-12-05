from typing import Dict, Any, Optional
from enum import Enum
import logging


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
    
    @staticmethod
    def _safe_str(value: Any) -> str:
        """Safely convert value to string, return empty string if None."""
        if value is None:
            return ''
        return str(value)
    
    def calculate_facet_rewards(
        self, 
        paper: Dict[str, Any],
        query_facets: Dict[str, Any],
        semantic_sim: float
    ) -> Dict[str, float]:
        """Calculate rewards for each query facet with robust None handling."""
        rewards = {
            'semantic': semantic_sim * 1.0,
            'temporal': 0.0,
            'institutional': 0.0,
            'intent': 0.0,
            'venue': 0.0,
        }
        
        # Temporal match
        if query_facets.get('temporal'):
            year = paper.get('year')
            if year:
                try:
                    temporal = query_facets['temporal']
                    if isinstance(temporal, (tuple, list)) and len(temporal) == 2:
                        start, end = temporal
                        if start is not None and end is not None and start <= year <= end:
                            rewards['temporal'] = 0.5
                except (TypeError, ValueError, AttributeError):
                    pass
        
        # Venue match
        if query_facets.get('venue'):
            venue = self._safe_str(paper.get('publication_name') or paper.get('venue'))
            query_venue = self._safe_str(query_facets['venue'])
            
            if venue and query_venue and query_venue.lower() in venue.lower():
                rewards['venue'] = 0.8
        
        # Intent match
        if query_facets.get('intent'):
            title = self._safe_str(paper.get('title')).lower()
            intent = self._safe_str(query_facets['intent'])
            
            if title and intent:
                if intent == 'survey' and ('survey' in title or 'review' in title):
                    rewards['intent'] = 0.7
                elif intent == 'methodology' and ('method' in title or 'algorithm' in title):
                    rewards['intent'] = 0.6
                elif intent == 'application' and ('application' in title or 'applied' in title):
                    rewards['intent'] = 0.5
                elif intent == 'theory' and ('theory' in title or 'theoretical' in title):
                    rewards['intent'] = 0.6
                elif intent == 'empirical' and ('experiment' in title or 'empirical' in title):
                    rewards['intent'] = 0.6
        
        return rewards

