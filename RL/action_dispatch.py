from typing import Dict, Tuple, Any, List
import logging
from graph.database import store

class RelationType:
    CITES = 0           # Get references (outgoing)
    CITED_BY = 1        # Get citations (incoming)
    WROTE = 2           # Paper -> Author
    AUTHORED = 3        # Author -> Paper
    COLLAB = 5          # Author -> Co-Authors
    KEYWORD_JUMP = 6    # Paper -> Similar Papers (via Keyword)
    VENUE_JUMP = 7      # Paper -> Similar Papers (via Venue/Publication)
    OLDER_REF = 8       # Paper -> Older References
    NEWER_CITED_BY = 9  # Paper -> Newer Citations
    SECOND_COLLAB = 10  # Author -> 2nd Degree Collaborators

# Maps (ActionID, NodeType) -> Function Name in store.py
ACTION_DISPATCH: Dict[Tuple[int, str], str] = {
    (RelationType.CITES, "Paper"): "get_references_by_paper",
    (RelationType.CITED_BY, "Paper"): "get_citations_by_paper",
    (RelationType.WROTE, "Paper"): "get_authors_by_paper_id", # Switched to ID version for safety
    (RelationType.AUTHORED, "Author"): "get_papers_by_author_id",
    (RelationType.COLLAB , "Author"): "get_collabs_by_author", 
    (RelationType.KEYWORD_JUMP, "Paper"): "get_papers_by_keyword",
    (RelationType.VENUE_JUMP, "Paper"): "get_papers_by_venue", 
    (RelationType.OLDER_REF, "Paper"): "get_older_references", 
    (RelationType.NEWER_CITED_BY , "Paper"): "get_newer_citations",
    (RelationType.SECOND_COLLAB, "Author"): "get_second_degree_collaborators"
}

async def execute_action(action_type: int, current_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes the store function corresponding to the action_type, 
    automatically extracting the correct argument from current_node.
    """
    
    # 1. Determine Node Type
    node_type = "Paper" if "paper_id" in current_node else "Author"
    
    # 2. Get Function Name
    func_name = ACTION_DISPATCH.get((action_type, node_type))
    if not func_name:
        logging.warning(f"Invalid Action {action_type} for Node Type {node_type}")
        return []
    
    # 3. Get the Function Object
    func = getattr(store, func_name, None)
    if not func:
        logging.error(f"Store function {func_name} not found.")
        return []

    # 4. Argument Resolution Strategy
    # We dynamically pick the right ID/property from the current node
    try:
        if action_type == RelationType.KEYWORD_JUMP:
            # Heuristic: Pick the first keyword if multiple exist
            keywords = current_node.get("keywords", "")
            if not keywords: return []
            # Assuming keywords are comma-separated or list. Take first meaningful one.
            keyword = keywords.split(',')[0].strip() if isinstance(keywords, str) else keywords[0]
            return await func(keyword, limit=5, exclude_paper_id=current_node.get("paper_id"))

        elif action_type == RelationType.VENUE_JUMP:
            venue = current_node.get("publication_name") or current_node.get("venue")
            if not venue: return []
            return await func(venue, exclude_paper_id=current_node.get("paper_id"))

        elif node_type == "Paper":
            # Most paper actions just need paper_id
            return await func(current_node["paper_id"])

        elif node_type == "Author":
            # Most author actions just need author_id
            return await func(current_node["author_id"])

    except Exception as e:
        logging.error(f"Error executing action {func_name}: {e}")
        return []

    return []