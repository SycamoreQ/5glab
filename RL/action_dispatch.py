from typing import Dict, Callable, Tuple

class RelationType:
    CITES = 0
    CITED_BY = 1
    WROTE = 2
    AUTHORED = 3
    SELF = 4
    COLLAB = 5 

ACTION_DISPATCH: Dict[Tuple[int, str], str] = {
    # (action, node_type): 'store_function_name'
    (RelationType.CITES, "Paper"): "get_references_by_paper",
    (RelationType.CITED_BY, "Paper"): "get_citations_by_paper",
    (RelationType.WROTE, "Paper"): "get_authors_by_paper",
    (RelationType.AUTHORED, "Author"): "get_papers_by_author",
    (RelationType.COLLAB , "Author"): "get_collabs_by_author", 
    
}

ACTION_ARG_MAP: Dict[Tuple[int, str], str] = {
    (RelationType.CITES, "Paper"): "paper_id",
    (RelationType.CITED_BY, "Paper"): "paper_id",
    (RelationType.WROTE, "Paper"): "title",
    (RelationType.AUTHORED, "Author"): "name",
}
