import dspy
from typing import Dict, Any, List, Optional , Literal
from datetime import datetime
from pydantic import BaseModel, Field
from unified import UnifiedQueryParser


class QueryFacets(BaseModel):
    """Enhanced structured output for parsed query."""
    
    semantic: str = Field(description="Core research topic/keywords")
    temporal: Optional[List[int]] = Field(
        default=None, 
        description="Year range as [start_year, end_year] or null"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key technical terms/topics"
    )
    
    author: Optional[str] = Field(
        default=None,
        description="Author name if mentioned"
    )
    author_search_mode: bool = Field(
        default=False,
        description="True if query asks for author's work"
    )
    institutional: Optional[str] = Field(
        default=None,
        description="University/company affiliation"
    )
    
    venue: Optional[str] = Field(
        default=None,
        description="Conference/journal name"
    )
    
    intent: Optional[str] = Field(
        default=None,
        description="One of: survey, methodology, application, theory, empirical, author_works, paper_navigation"
    )
    
    paper_title: Optional[str] = Field(
        default=None,
        description="Specific paper title if mentioned (e.g., 'paper titled XYZ')"
    )
    
    paper_search_mode: bool = Field(
        default=False,
        description="True if query is about a specific paper (not a topic search)"
    )
    
    paper_operation: Optional[Literal[
        'citations',       # Papers citing this paper (CITED_BY)
        'references',      # Papers cited by this paper (CITES)
        'related',         # Similar papers
        'coauthors',       # Authors of this paper
        'venue_papers',    # Other papers in same venue
        'find_paper'       # Just finding the paper itself
    ]] = Field(
        default=None,
        description="Specific operation on a paper"
    )
    
    relation_focus: Optional[Literal[
        'CITES',           # Focus on reference relationships
        'CITED_BY',        # Focus on citation relationships
        'WROTE',           # Focus on authorship
        'COLLAB',          # Focus on collaborations
        'PUBLISHED_IN',    # Focus on venue
        'SAME_COMMUNITY'   # Focus on research communities
    ]] = Field(
        default=None,
        description="Specific relationship type to explore"
    )
    
    hop_depth: Optional[int] = Field(
        default=None,
        description="Number of hops to traverse (e.g., 'second-order citations')"
    )



class ParseResearchQuery(dspy.Signature): 
    query = dspy.InputField(desc = "Natural Language Research Query")
    facets = dspy.OutputField(
        desc = "Structured Query Facets as JSON with paper specific fields",
        type = QueryFacets
    )


class OptimQueryParser(dspy.Module): 

    def __init__(self , model:str = 'llama3.2' , use_optimized: bool = True ): 
        super().__init__()
        lm = dspy.LM(model , max_tokens=400)
        dspy.configure(lm = lm )
        self.parse = dspy.Predict(ParseResearchQuery)
        self.current_year = datetime.now().year 

    def forward(self, query: str) -> QueryFacets:
        """Parse query with paper-specific handling."""
        
        enhanced_query = f"{query} [Current year: {self.current_year}]"
        
        result = self.parse(query=enhanced_query)
        
        return result.facets
    
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse query and return dict (for unified interface)."""
        facets = self.forward(query)
        
        # Convert Pydantic model to dict
        if hasattr(facets, 'dict'):
            result = facets.dict()
        elif hasattr(facets, 'model_dump'):
            result = facets.model_dump()
        else:
            result = dict(facets)
        
        result['original'] = query
        
        # Convert temporal list to tuple
        if result.get('temporal') and isinstance(result['temporal'], list):
            result['temporal'] = tuple(result['temporal'])
        
        return result
    

def create_training_examples() -> List[dspy.Example]:
    """Enhanced examples including paper-specific queries."""
    
    examples = [
        dspy.Example(
            query="what is the recent publication by author XYZ",
            facets=QueryFacets(
                semantic="publications",
                temporal=[2021, 2024],
                author="XYZ",
                author_search_mode=True,
                intent="author_works"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="deep learning papers on medical imaging from Stanford 2020-2023",
            facets=QueryFacets(
                semantic="deep learning medical imaging",
                temporal=[2020, 2023],
                institutional="Stanford",
                intent="application",
                entities=["deep learning", "medical imaging"]
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="Get me the citations of this paper titled Attention Is All You Need",
            facets=QueryFacets(
                semantic="",  # Not a topic search
                paper_title="Attention Is All You Need",
                paper_search_mode=True,
                paper_operation="citations",
                relation_focus="CITED_BY",
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="what papers does the BERT paper cite",
            facets=QueryFacets(
                semantic="",
                paper_title="BERT",
                paper_search_mode=True,
                paper_operation="references",
                relation_focus="CITES",
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="show me papers similar to AlexNet",
            facets=QueryFacets(
                semantic="convolutional neural networks image classification",  # Infer topic
                paper_title="AlexNet",
                paper_search_mode=True,
                paper_operation="related",
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="who are the authors of ImageNet Classification with Deep Convolutional Neural Networks",
            facets=QueryFacets(
                semantic="",
                paper_title="ImageNet Classification with Deep Convolutional Neural Networks",
                paper_search_mode=True,
                paper_operation="coauthors",
                relation_focus="WROTE",
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="find papers that cite ResNet and are published in CVPR",
            facets=QueryFacets(
                semantic="",
                paper_title="ResNet",
                paper_search_mode=True,
                paper_operation="citations",
                venue="CVPR",
                relation_focus="CITED_BY",
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="what are the second-order citations of the GPT-3 paper",
            facets=QueryFacets(
                semantic="",
                paper_title="GPT-3",
                paper_search_mode=True,
                paper_operation="citations",
                relation_focus="CITED_BY",
                hop_depth=2,
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="references of Transformer paper from 2020 onwards",
            facets=QueryFacets(
                semantic="",
                paper_title="Transformer",
                paper_search_mode=True,
                paper_operation="references",
                relation_focus="CITES",
                temporal=[2020, 2024],
                intent="paper_navigation"
            )
        ).with_inputs("query"),
        
        dspy.Example(
            query="papers on attention mechanisms that cite the Transformer paper",
            facets=QueryFacets(
                semantic="attention mechanisms",
                paper_title="Transformer",
                paper_search_mode=True, 
                paper_operation="citations",
                relation_focus="CITED_BY",
                entities=["attention mechanisms"],
                intent="paper_navigation"
            )
        ).with_inputs("query"),
    

        dspy.Example(
            query = "who has author XYZ collaborted with in the year 2020?",
            facets = QueryFacets(
                semantic="",
                temporal=[2020],
                author="XYZ",
                relation_focus= "COLLAB",
                intent="author_collab",
            )
        ).with_inputs("query"), 
    ]
    
    return examples



def parsing_metric(example , prediction , trace = None) -> float: 
    gold = example.facets
    pred = prediction.facets 

    score = 0.0
    total_fields = 0

    fields = [
        'semantic', 'author', 'author_search_mode', 'institutional', 
        'venue', 'intent', 'paper_title', 'paper_search_mode', 
        'paper_operation', 'relation_focus'
    ]
    

