from typing import Dict, Tuple, Any, List
import logging
from graph.database import store

from typing import Dict, Tuple, Any, List
import logging
import asyncio

class RelationType:
    CITES = 0           # Paper: Get references (outgoing)
    CITED_BY = 1        # Paper: Get citations (incoming)
    WROTE = 2           # Paper -> Author
    AUTHORED = 3        # Author -> Paper
    COLLAB = 5          # Author -> Co-Authors
    KEYWORD_JUMP = 6    # Paper: Similar Papers (via Keyword)
    VENUE_JUMP = 7      # Paper: Similar Papers (via Venue/Publication)
    OLDER_REF = 8       # Paper: Older References
    NEWER_CITED_BY = 9  # Paper: Newer Citations
    SECOND_COLLAB = 10  # Author: 2nd Degree Collaborators
    STOP = 11           # Stops further exploration 
    INFLUENCE_PATH = 12 

ACTION_SPACE_MAP = {
    RelationType.CITES: "get_references_by_paper",
    RelationType.CITED_BY: "get_citations_by_paper",
    RelationType.WROTE: "get_authors_by_paper_id",
    RelationType.AUTHORED: "get_papers_by_author_id",
    RelationType.COLLAB: "get_collabs_by_author",
    RelationType.KEYWORD_JUMP: "get_papers_by_keyword",
    RelationType.VENUE_JUMP: "get_papers_by_venue",
    RelationType.OLDER_REF: "get_older_references",
    RelationType.NEWER_CITED_BY: "get_newer_citations",
    RelationType.SECOND_COLLAB: "get_second_degree_collaborators",
    RelationType.INFLUENCE_PATH: "get_influence_path_papers",
    RelationType.STOP: "stop_action",
}

ACTION_VALID_SOURCE_TYPE: Dict[int, str] = {
    RelationType.CITES: "Paper",
    RelationType.CITED_BY: "Paper",
    RelationType.WROTE: "Paper",
    RelationType.AUTHORED: "Author",
    RelationType.COLLAB: "Author",
    RelationType.KEYWORD_JUMP: "Paper",
    RelationType.VENUE_JUMP: "Paper",
    RelationType.OLDER_REF: "Paper",
    RelationType.NEWER_CITED_BY: "Paper",
    RelationType.SECOND_COLLAB: "Author",
    RelationType.INFLUENCE_PATH: "Author",
    RelationType.STOP: "Any",
}

async def execute_action(action_type: int, current_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes the store function corresponding to the action_type, 
    automatically extracting the correct argument from current_node.
    """
    node_type = "Paper" if "paper_id" in current_node else "Author"
    
    func_name = ACTION_DISPATCH.get((action_type, node_type))
    if not func_name:
        logging.warning(f"Invalid Action {action_type} for Node Type {node_type}")
        return []
    
    func = getattr(store, func_name, None)
    if not func:
        logging.error(f"Store function {func_name} not found.")
        return []
    try:
        if action_type == RelationType.KEYWORD_JUMP:
            keywords = current_node.get("keywords", "")
            if not keywords: return []
            keyword = keywords.split(',')[0].strip() if isinstance(keywords, str) else keywords[0]
            return await func(keyword, limit=5, exclude_paper_id=current_node.get("paper_id"))

        elif action_type == RelationType.VENUE_JUMP:
            venue = current_node.get("publication_name") or current_node.get("venue")
            if not venue: return []
            return await func(venue, exclude_paper_id=current_node.get("paper_id"))

        elif node_type == "Paper":
            return await func(current_node["paper_id"])

        elif node_type == "Author":
            return await func(current_node["author_id"])

    except Exception as e:
        logging.error(f"Error executing action {func_name}: {e}")
        return []

    return []