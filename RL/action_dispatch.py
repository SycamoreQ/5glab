from typing import Dict, Callable, Tuple

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

ACTION_DISPATCH: Dict[Tuple[int, str], str] = {
    # (action, node_type): 'store_function_name'
    (RelationType.CITES, "Paper"): "get_references_by_paper",
    (RelationType.CITED_BY, "Paper"): "get_citations_by_paper",
    (RelationType.WROTE, "Paper"): "get_authors_by_paper",
    (RelationType.AUTHORED, "Author"): "get_papers_by_author",
    (RelationType.COLLAB , "Author"): "get_collabs_by_author", 
    (RelationType.KEYWORD_JUMP, "Paper"): "get_papers_by_keyword",
    (RelationType.VENUE_JUMP, "Paper"):"get_papers_by_venue", 
    (RelationType.OLDER_REF, "Paper"): "get_older_references", 
    (RelationType.NEWER_CITED_BY , "Paper"): "get_newer_citations"
}

ACTION_ARG_MAP: Dict[Tuple[int, str], str] = {
    (RelationType.CITES, "Paper"): "paper_id",
    (RelationType.CITED_BY, "Paper"): "paper_id",
    (RelationType.WROTE, "Paper"): "title",
    (RelationType.AUTHORED, "Author"): "name",
    (RelationType.COLLAB, "Author"): "author_id",
    (RelationType.KEYWORD_JUMP, "Paper"): "keyword",
    (RelationType.VENUE_JUMP, "Paper") : "venue", 
    (RelationType.OLDER_REF , "Paper"): "paper_id",  
    (RelationType.NEWER_CITED_BY , "Paper"): "paper_id"
}
