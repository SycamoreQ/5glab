import kuzu 
import logging
import os 
from kuzu import Database, Connection
import datetime 

DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)


def get_papers_by_author(author_name):
    """Get all papers by a specific author"""
    try:
        result = conn.execute("""
            MATCH (a:Author {name: $name})-[:WROTE]->(p:Paper)
            RETURN p.title, p.year, p.doi, p.paper_id
            """, {"name": author_name})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]


def get_authors_by_paper(paper_title):
    """Get all authors of a specific paper"""

    try:
        result = conn.execute("""
            MATCH (a:Author)-[:WROTE]->(p:Paper {title: $title})
            RETURN a.name, a.author_id
            """, {"title": paper_title})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_paper_by_year(year):
    """Get all papers published in a specific year"""

    try:
        result = conn.execute("""
            MATCH (p:Paper {year: $year})
            RETURN p.title, p.doi, p.paper_id
            """, {"year": year})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_papers_by_year_range(start_year, end_year):
    """Get all papers within a year range"""
    try:
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE p.year >= $start_year AND p.year <= $end_year
            RETURN p.title, p.year, p.doi, p.paper_id
            ORDER BY p.year DESC
            """, {"start_year": start_year, "end_year": end_year})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_paper_by_id(paper_id):
    """Get a specific paper by its ID"""
    try:
        result = conn.execute("""
            MATCH (p:Paper {paper_id: $id})
            RETURN p.title, p.year, p.doi, p.paper_id
            """, {"id": paper_id})
        papers = [dict(row) for row in result]
        return papers[0] if papers else None
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return None



def get_paper_by_doi(doi):
    """Get a specific paper by its DOI"""
    try:
        result = conn.execute("""
            MATCH (p:Paper {doi: $doi})
            RETURN p.title, p.year, p.paper_id
            """, {"doi": doi})
        papers = [dict(row) for row in result]
        return papers[0] if papers else None
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return None



def get_author_by_id(author_id):
    """Get a specific author by their ID"""

    try:
        result = conn.execute("""
            MATCH (a:Author {author_id: $id})
            RETURN a.name, a.author_id
            """, {"id": author_id})
        authors = [dict(row) for row in result]
        return authors[0] if authors else None
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return None




def search_papers_by_title(title_substring):
    """Search papers by title (case-insensitive substring match)"""

    try:
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS toLower($substring)
            RETURN p.title, p.year, p.doi, p.paper_id
            """, {"substring": title_substring})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def search_authors_by_name(name_substring):
    """Search authors by name (case-insensitive substring match)"""
    try:
        result = conn.execute("""
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($substring)
            RETURN a.name, a.author_id
            """, {"substring": name_substring})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_all_papers(limit=None):
    """Get all papers, optionally with a limit"""
    query = "MATCH (p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    params = {}
    
    if limit:
        query += " LIMIT $limit"
        params["limit"] = limit
        
    try:
        result = conn.execute(query, params)
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_all_authors(limit=None):
    """Get all authors, optionally with a limit"""
    query = "MATCH (a:Author) RETURN a.name, a.author_id ORDER BY a.name"
    params = {}
    
    if limit:
        query += " LIMIT $limit"
        params["limit"] = limit
        
    try:
        result = conn.execute(query, params)
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]


def regex_for_paper(regex_string: str , limit=100):
    """Perform regex search on paper titles"""
    try: 
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE p.title =~ $regex
            RETURN p.title
            LIMIT $limit
            """, {"regex": regex_string, "limit": limit})
        
    except Exception as e: 
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]


def regex_for_author(regex_string: str , limit=100):
    """Perform regex search on paper titles"""
    try: 
        result = conn.execute("""
            MATCH (a:Author)
            WHERE a.title =~ $regex
            RETURN a.title
            LIMIT $limit
            """, {"regex": regex_string, "limit": limit})
        
    except Exception as e: 
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



# update Functions
def update_author(author_id, new_name=None):
    """Update author information"""
    if not new_name:
        return None
        
    try:
        result = conn.execute("""
            MATCH (a:Author {author_id: $author_id})
            SET a.name = $new_name
            RETURN a.name, a.author_id
            """, {"author_id": author_id, "new_name": new_name})
        updated = [dict(row) for row in result]
        return updated[0] if updated else None
    except Exception as e:
        logging.error(f"Update author failed: {e}")
        return None

def update_paper(paper_id, title=None, year=None, doi=None):
    """Update paper information"""
    set_clauses = []
    params = {"paper_id": paper_id}
    
    if title:
        set_clauses.append("p.title = $title")
        params["title"] = title
    if year:
        set_clauses.append("p.year = $year")  
        params["year"] = year
    if doi:
        set_clauses.append("p.doi = $doi")
        params["doi"] = doi
        
    if not set_clauses:
        return None
        
    query = f"""
        MATCH (p:Paper {{paper_id: $paper_id}})
        SET {', '.join(set_clauses)}
        RETURN p.title, p.year, p.doi, p.paper_id
    """
    
    try:
        result = conn.execute(query, params)
        updated = [dict(row) for row in result]
        return updated[0] if updated else None
    except Exception as e:
        logging.error(f"Update paper failed: {e}")
        return None



#delete functions
def delete_author(author_id):
    """Delete an author and their relationships"""
    try:
        result = conn.execute("""
            MATCH (a:Author {author_id: $author_id})
            DETACH DELETE a
            """, {"author_id": author_id})
        return True
    except Exception as e:
        logging.error(f"Delete author failed: {e}")
        return False
    


def delete_paper(paper_id):
    """Delete a paper and its relationships"""
    try:
        result = conn.execute("""
            MATCH (p:Paper {paper_id: $paper_id})
            DETACH DELETE p
            """, {"paper_id": paper_id})
        return True
    except Exception as e:
        logging.error(f"Delete paper failed: {e}")
        return False
    


def delete_authorship(author_id, paper_id):
    """Delete a specific WROTE relationship"""
    try:
        result = conn.execute("""
            MATCH (a:Author {author_id: $author_id})-[r:WROTE]->(p:Paper {paper_id: $paper_id})
            DELETE r
            """, {"author_id": author_id, "paper_id": paper_id})
        return True
    except Exception as e:
        logging.error(f"Delete authorship failed: {e}")
        return False




# ANALYTICS Functions
def get_author_collaboration_count(author_id):
    """Get number of unique collaborators for an author"""
    try:
        result = conn.execute("""
            MATCH (a1:Author {author_id: $author_id})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
            WHERE a1 <> a2
            RETURN count(DISTINCT a2) as collaborator_count
            """, {"author_id": author_id})
        counts = [dict(row) for row in result]
        return counts[0]['collaborator_count'] if counts else 0
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return 0

def get_most_prolific_authors(limit=10):
    """Get authors with most papers"""
    try:
        result = conn.execute("""
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            RETURN a.name, a.author_id, count(p) as paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
            """, {"limit": limit})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]

def get_papers_per_year():
    """Get paper count by year"""
    try:
        result = conn.execute("""
            MATCH (p:Paper)
            RETURN p.year, count(p) as paper_count
            ORDER BY p.year DESC
            """)
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]

#citation Functions
def get_citations_by_paper(paper_id):
    """Get all papers that cite a specific paper"""
    try:
        result = conn.execute("""
            MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $paper_id})
            RETURN citing.title, citing.year, citing.doi, citing.paper_id
            ORDER BY citing.year DESC
            """, {"paper_id": paper_id})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_references_by_paper(paper_id):
    """Get all papers referenced/cited by a specific paper"""

    try:
        result = conn.execute("""
            MATCH (citing:Paper {paper_id: $paper_id})-[:CITES]->(cited:Paper)
            RETURN cited.title, cited.year, cited.doi, cited.paper_id
            ORDER BY cited.year DESC
            """, {"paper_id": paper_id})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]




def get_citation_count(paper_id):
    """Get citation count for a specific paper"""

    try:
        result = conn.execute("""
            MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $paper_id})
            RETURN count(citing) as citation_count
            """, {"paper_id": paper_id})
        counts = [dict(row) for row in result]
        return counts[0]['citation_count'] if counts else 0
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return 0
    

def get_most_cited_papers(limit=10):
    """Get papers with highest citation counts"""

    try:
        result = conn.execute("""
            MATCH (citing:Paper)-[:CITES]->(cited:Paper)
            RETURN cited.title, cited.year, cited.doi, cited.paper_id, count(citing) as citation_count
            ORDER BY citation_count DESC
            LIMIT $limit
            """, {"limit": limit})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_citation_depth(paper_id, max_depth=3):
    """Get citation network up to specified depth"""

    try:
        result = conn.execute("""
            MATCH path = (start:Paper {paper_id: $paper_id})-[:CITES*1..$max_depth]->(end:Paper)
            RETURN end.title, end.year, end.paper_id, length(path) as depth
            ORDER BY depth, end.year DESC
            """, {"paper_id": paper_id, "max_depth": max_depth})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def get_co_citation_papers(paper_id, limit=10):
    """Get papers that are co-cited with the given paper"""

    try:
        result = conn.execute("""
            MATCH (p1:Paper {paper_id: $paper_id})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
            WHERE p1 <> p2
            RETURN p2.title, p2.year, p2.paper_id, count(citing) as co_citation_count
            ORDER BY co_citation_count DESC
            LIMIT $limit
            """, {"paper_id": paper_id, "limit": limit})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def detect_research_communities(min_cluster_size=5):
    """Detect research communities using citation and collaboration patterns"""

    try:
        result = conn.execute("""
            MATCH (a1:Author)-[:WROTE]->(p1:Paper)-[:CITES]->(p2:Paper)<-[:WROTE]-(a2:Author)
            WHERE a1 <> a2
            WITH a1, a2, count(*) as connection_strength
            WHERE connection_strength >= 2
            RETURN a1.name, a2.name, connection_strength
            ORDER BY connection_strength DESC
            """)
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]



def track_keyword_citation_impact(keyword):
    """Track how papers containing a keyword perform in citations over time"""

    try:
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS $keyword 
                  OR (p.abstract IS NOT NULL AND toLower(p.abstract) CONTAINS $keyword)
            
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            WITH p, count(citing) as citation_count
            
            RETURN p.year, 
                   count(p) as papers_with_keyword,
                   avg(citation_count) as avg_citations_per_paper,
                   sum(citation_count) as total_citations,
                   max(citation_count) as max_citations
            ORDER BY p.year
            """, {"keyword": keyword.lower()})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]
    


def track_keyword_temporal_trend(keyword, start_year=None, end_year=None):
    """Track how a specific keyword's usage evolves over time"""
    try:
        year_filter = ""
        params = {"keyword": keyword.lower()}
        
        if start_year:
            year_filter += "AND p.year >= $start_year "
            params["start_year"] = start_year
        if end_year:
            year_filter += "AND p.year <= $end_year "
            params["end_year"] = end_year
            
        result = conn.execute(f"""
            MATCH (p:Paper)
            WHERE (toLower(p.title) CONTAINS $keyword 
                   OR (p.keywords IS NOT NULL AND toLower(p.keywords) CONTAINS $keyword))
                  {year_filter}
            RETURN p.year, count(p) as keyword_count
            ORDER BY p.year
            """, params)
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return []
    return [dict(row) for row in result]