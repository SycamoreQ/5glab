import kuzu 
import logging
import os 
from kuzu import Database, Connection

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


def path()