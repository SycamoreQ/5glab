import dspy
from typing import List, Optional, Literal, Any, Dict, Tuple
from pydantic import BaseModel, Field


class Constraint(BaseModel):
    """Single constraint on entities."""
    field: str
    operator: Literal['equals', 'contains', 'greater_than', 'less_than', 'between', 'in_list']
    value: Any
    
    def __str__(self):
        return f"{self.field} {self.operator} {self.value}"

class QueryIntent(BaseModel):
    """Hierarchical query representation."""
    target_entity: Literal['papers', 'authors', 'venues', 'collaborations', 'communities']
    operation: Literal[
        'find', 'citations', 'references', 'authors', 'papers', 
        'collaborators', 'related', 'count', 'traverse'
    ] = 'find'
    semantic: str = ""
    constraints: List[Constraint] = Field(default_factory=list)

    author: Optional[str] = None
    paper_title: Optional[str] = None
    venue: Optional[str] = None
    temporal: Optional[List[int]] = None
    field_of_study: Optional[str] = None
    min_citation_count: Optional[int] = None
    max_citation_count: Optional[int] = None
    min_author_count: Optional[int] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class ParseComplexQuery(dspy.Signature):
    """Parse hierarchical research query into structured format.
    
    IMPORTANT INSTRUCTIONS:
    - target_entity: MUST be one of: papers, authors, venues, collaborations, communities
    - operation: MUST be one of: find, citations, references, authors, papers, collaborators, related
    - semantic: ONLY the core research topic (remove ALL constraints like venue, year, author names, citation counts)
    - constraints_json: JSON array of constraints. Each constraint MUST have:
      * field: venue, field_of_study, author, paper_title, citation_count, year, author_count
      * operator: MUST be exactly one of: equals, contains, greater_than, less_than, between, in_list
      * value: the constraint value
    
    EXAMPLES:
    
    Query: "Get me authors who wrote papers in 'IEEJ Transactions' in the field of Physics"
    target_entity: authors
    operation: find
    semantic: Physics
    constraints_json: [{"field": "venue", "operator": "contains", "value": "IEEJ Transactions"}, {"field": "field_of_study", "operator": "contains", "value": "Physics"}]
    
    Query: "papers on transformers with more than 100 citations from 2020-2023"
    target_entity: papers
    operation: find
    semantic: transformers
    constraints_json: [{"field": "citation_count", "operator": "greater_than", "value": 100}, {"field": "year", "operator": "between", "value": [2020, 2023]}]
    
    Query: "deep learning papers"
    target_entity: papers
    operation: find
    semantic: deep learning
    constraints_json: []
    """
    
    query: str = dspy.InputField(desc="User's natural language research query")
    
    # Output fields
    target_entity: str = dspy.OutputField(desc="MUST be one of: papers, authors, venues, collaborations, communities")
    operation: str = dspy.OutputField(desc="MUST be one of: find, citations, references, authors, papers, collaborators, related")
    semantic: str = dspy.OutputField(desc="Core topic ONLY (no constraints)")
    constraints_json: str = dspy.OutputField(desc='JSON array: [{"field": "venue", "operator": "contains", "value": "..."}]. Operator MUST be: equals, contains, greater_than, less_than, between, in_list')



class HierarchicalQueryParser(dspy.Module):
    """Modular hierarchical parser using DSPy with Ollama."""
    
    def __init__(self, use_cot: bool = True):
        super().__init__()
        

        if use_cot:
            self.parse = dspy.ChainOfThought(ParseComplexQuery)
        else:
            self.parse = dspy.Predict(ParseComplexQuery)

    def _normalize_operator(self, op: str) -> str:
        op = op.lower().strip()
    
        operator_map = {
            'eq': 'equals',
            '==': 'equals',
            'equal': 'equals',
            'equals': 'equals',
            
            'contains': 'contains',
            'include': 'contains',
            'includes': 'contains',
            'in': 'contains',
            
            'gt': 'greater_than',
            '>': 'greater_than',
            'greater': 'greater_than',
            'greater_than': 'greater_than',
            'more_than': 'greater_than',
            
            'lt': 'less_than',
            '<': 'less_than',
            'less': 'less_than',
            'less_than': 'less_than',
            
            'between': 'between',
            'range': 'between',
            
            'in_list': 'in_list',
            'one_of': 'in_list',
        }
        
        return operator_map.get(op, 'contains') 
    
    def forward(self, query: str) -> QueryIntent:
        """Parse query into QueryIntent."""
        try:
            result = self.parse(query=query)
            
            target_entity = result.target_entity.lower().strip()
            operation = result.operation.lower().strip()
            semantic = result.semantic.strip()
            
            constraints = []
            try:
                import json
                constraints_str = result.constraints_json.strip()
                if constraints_str.startswith('```'):
                    constraints_str = constraints_str.split('```')[1]
                    if constraints_str.startswith('json'):
                        constraints_str = constraints_str[4:]
                constraints_str = constraints_str.strip()
                
                constraints_data = json.loads(constraints_str)
                for c in constraints_data:
                    constraints.append(Constraint(**c))
            except Exception as e:
                print(f"[WARN] Constraint parsing failed: {e}")
                pass
            
            # Validate target_entity
            valid_targets = ['papers', 'authors', 'venues', 'collaborations', 'communities']
            if target_entity not in valid_targets:
                target_entity = 'papers'
            
            # Validate operation
            valid_ops = ['find', 'citations', 'references', 'authors', 'papers', 'collaborators', 'related', 'count', 'traverse']
            if operation not in valid_ops:
                operation = 'find'
            
            # Build QueryIntent
            intent = QueryIntent(
                target_entity=target_entity,
                operation=operation,
                semantic=semantic,
                constraints=constraints
            )
            
            # Extract legacy fields
            for constraint in constraints:
                if constraint.field == 'venue':
                    intent.venue = str(constraint.value)
                elif constraint.field == 'author':
                    intent.author = str(constraint.value)
                elif constraint.field == 'paper_title':
                    intent.paper_title = str(constraint.value)
                elif constraint.field == 'field_of_study':
                    intent.field_of_study = str(constraint.value)
                elif constraint.field == 'year' and constraint.operator == 'between':
                    intent.temporal = constraint.value
                elif constraint.field == 'citation_count' and constraint.operator == 'greater_than':
                    intent.min_citation_count = int(constraint.value)
            
            return intent
            
        except Exception as e:
            print(f"[ERROR] DSPy parsing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            return QueryIntent(
                target_entity='papers',
                operation='find',
                semantic=query,
                constraints=[]
            )

def create_hierarchical_examples() -> List[dspy.Example]:
    """Create training examples for DSPy optimization."""
    
    examples = [
        dspy.Example(
            query="deep learning papers",
            target_entity='papers',
            operation='find',
            semantic='deep learning',
            constraints_json='[]'
        ).with_inputs('query'),
        
        dspy.Example(
            query="Get me authors who wrote papers in 'IEEJ Transactions' in the field of Physics",
            target_entity='authors',
            operation='find',
            semantic='Physics',
            constraints_json='[{"field": "venue", "operator": "contains", "value": "IEEJ Transactions"}, {"field": "field_of_study", "operator": "contains", "value": "Physics"}]'
        ).with_inputs('query'),
        
        dspy.Example(
            query="papers on transformers with more than 100 citations from 2020-2023",
            target_entity='papers',
            operation='find',
            semantic='transformers',
            constraints_json='[{"field": "citation_count", "operator": "greater_than", "value": 100}, {"field": "year", "operator": "between", "value": [2020, 2023]}]'
        ).with_inputs('query'),
        
        dspy.Example(
            query="papers that cite BERT",
            target_entity='papers',
            operation='citations',
            semantic='',
            constraints_json='[{"field": "paper_title", "operator": "equals", "value": "BERT"}]'
        ).with_inputs('query'),
        
        dspy.Example(
            query="collaborators of Yann LeCun who published in NeurIPS",
            target_entity='authors',
            operation='collaborators',
            semantic='',
            constraints_json='[{"field": "author", "operator": "equals", "value": "Yann LeCun"}, {"field": "venue", "operator": "contains", "value": "NeurIPS"}]'
        ).with_inputs('query'),
    ]
    
    return examples

class DSPyHierarchicalParser:
    """
    DSPy-based parser with proper Ollama integration.
    """
    
    def __init__(self, model: str = "llama3.2", optimize: bool = False):
        try:
            lm = dspy.LM(
                model=f'ollama_chat/{model}',
                api_base='http://localhost:11434',
                temperature=0.1,
                max_tokens=600
            )
            dspy.configure(lm=lm)
            print(f"[DSPy] Configured with Ollama model: {model}")
            
        except Exception as e:
            print(f"[ERROR] DSPy LM configuration failed: {e}")
            print("[INFO] Falling back to direct ollama/ prefix...")
            
            try:
                lm = dspy.LM(
                    model=f'ollama/{model}',
                    api_base='http://localhost:11434',
                    temperature=0.1,
                    max_tokens=600
                )
                dspy.configure(lm=lm)
                print(f"[DSPy] Configured with fallback method")
            except Exception as e2:
                print(f"[ERROR] Both methods failed: {e2}")
                raise

        self.parser = HierarchicalQueryParser(use_cot=True)

        if optimize:
            print("[DSPy] Optimizing parser with examples...")
            examples = create_hierarchical_examples()
            
            try:
                from dspy.teleprompt import BootstrapFewShot
                
                def metric(example, pred, trace=None):
                    score = 0.0
                    if hasattr(pred, 'target_entity') and example.target_entity:
                        if pred.target_entity.lower() == example.target_entity.lower():
                            score += 50.0
                    if hasattr(pred, 'operation') and example.operation:
                        if pred.operation.lower() == example.operation.lower():
                            score += 50.0
                    return score
                
                optimizer = BootstrapFewShot(
                    metric=metric,
                    max_bootstrapped_demos=3,
                    max_labeled_demos=3
                )
                
                self.parser = optimizer.compile(
                    self.parser,
                    trainset=examples
                )
                print("[DSPy] Optimization complete!")
                
            except Exception as e:
                print(f"[WARN] Optimization failed: {e}. Using unoptimized parser.")
    
    def parse(self, query: str) -> QueryIntent:
        """Parse query using DSPy."""
        return self.parser(query=query)
    
    def to_dict(self, intent: QueryIntent) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'target_entity': intent.target_entity,
            'operation': intent.operation,
            'semantic': intent.semantic,
            'constraints': [
                {'field': c.field, 'operator': c.operator, 'value': c.value}
                for c in intent.constraints
            ],
            'author': intent.author,
            'paper_title': intent.paper_title,
            'venue': intent.venue,
            'temporal': intent.temporal,
            'field_of_study': intent.field_of_study,
            'min_citation_count': intent.min_citation_count,
        }


class HierarchicalRewardMapper:
    """Enhanced reward mapper for hierarchical queries."""
    
    def __init__(self, config):
        self.config = config
    
    def get_manager_reward(
        self,
        relation_type: int,
        query_intent: QueryIntent,
        current_node: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Calculate manager reward."""
        from RL.env import RelationType
        
        reward = 0.0
        reasons = []
        
        if query_intent.operation == 'citations':
            if relation_type == RelationType.CITED_BY:
                reward += 25.0
                reasons.append("op:citations")
        elif query_intent.operation == 'references':
            if relation_type == RelationType.CITES:
                reward += 25.0
                reasons.append("op:references")
        elif query_intent.operation == 'authors':
            if relation_type == RelationType.WROTE:
                reward += 25.0
                reasons.append("op:authors")
        elif query_intent.operation == 'collaborators':
            if relation_type == RelationType.COLLAB:
                reward += 25.0
                reasons.append("op:collab")
        
        if query_intent.target_entity == 'authors':
            if relation_type in [RelationType.WROTE, RelationType.AUTHORED]:
                reward += 15.0
                reasons.append("target:authors")

        for constraint in query_intent.constraints:
            if constraint.field == 'venue' and relation_type == RelationType.VENUE_JUMP:
                reward += 15.0
                reasons.append(f"venue:{str(constraint.value)[:15]}")
            elif constraint.field == 'field_of_study' and relation_type == RelationType.KEYWORD_JUMP:
                reward += 12.0
                reasons.append(f"field:{str(constraint.value)[:15]}")
        
        return reward, " | ".join(reasons) if reasons else "none"
    
    def get_worker_reward(
    self,
    node: Dict[str, Any],
    query_intent: QueryIntent,
    semantic_sim: float
) -> Tuple[float, str]:
        """Calculate worker reward checking constraints."""
        reward = 0.0
        reasons = []
        
        paper_id = node.get('paper_id') or node.get('paperId')
        author_id = node.get('author_id') or node.get('authorId')
        
        apply_penalties = semantic_sim > 0.4
        
        for constraint in query_intent.constraints:
            if constraint.field == 'venue':
                node_venue = (node.get('venue', '') or node.get('publicationName', '')).lower()
                constraint_value = str(constraint.value).lower()
                
                if constraint.operator == 'contains':
                    if constraint_value in node_venue:
                        reward += 20.0
                        reasons.append(f"✓venue")
                    elif apply_penalties:  
                        reward -= 5.0  
                        reasons.append(f"✗venue")
            
            elif constraint.field == 'field_of_study':
                node_fields = node.get('fieldsOfStudy', []) or node.get('fields', [])
                if isinstance(node_fields, list):
                    match = any(str(constraint.value).lower() in str(f).lower() for f in node_fields)
                    if match:
                        reward += 18.0
                        reasons.append(f"✓field")
                    elif apply_penalties:
                        reward -= 4.0
                        reasons.append(f"✗field")
            
            elif constraint.field == 'citation_count':
                cit_count = node.get('citationCount', 0)
                if constraint.operator == 'greater_than':
                    if cit_count > constraint.value:
                        reward += 15.0
                        reasons.append(f"✓cit:{cit_count}")
                    elif apply_penalties:
                        reward -= 5.0  
                        reasons.append(f"✗cit:{cit_count}")
            
            elif constraint.field == 'year':
                paper_year = node.get('year')
                if constraint.operator == 'between' and paper_year:
                    start, end = constraint.value
                    if start <= paper_year <= end:
                        reward += 12.0
                        reasons.append(f"✓year:{paper_year}")
                    elif apply_penalties:
                        reward -= 4.0  
                        reasons.append(f"✗year:{paper_year}")
        
        if query_intent.constraints:
            passed = sum(1 for r in reasons if r.startswith('✓'))
            total = len(query_intent.constraints)
            if passed == total:
                reward += 30.0
                reasons.append(f"✓✓ALL({passed})")
        
        return reward, " | ".join(reasons) if reasons else "none"

