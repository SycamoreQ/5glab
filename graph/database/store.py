import kuzu
import logging
import asyncio
from kuzu import Database, Connection
from typing import Any, List, Optional, Dict

DB_PATH = "research_db"
# Initialize database and connection once. Be careful: the Kuzu Connection object
# may not be fully thread-safe. If you see odd behavior under heavy concurrency,
# consider creating a new Connection per request or using a connection pool. //TODO 
_db = Database(DB_PATH)
_conn = Connection(_db)


async def _run_query(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Run a Kuzu query in a thread to avoid blocking the event loop.

    Returns a list of dict rows.
    """
    params = params or []

    def _exec():
        try:
            result = _conn.execute(query, params)
            # Convert to list immediately while still in the thread.
            return [dict(row) for row in result]
        except Exception as e:
            logging.error(f"Query failed: {e}")
            raise

    rows = await asyncio.to_thread(_exec)
    return rows


# -- Read functions --
async def get_papers_by_author(author_name: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author {name: $1})-[:WROTE]->(p:Paper)
        RETURN p.title, p.year, p.doi, p.paper_id
        """
    try:
        return await _run_query(query, [author_name])
    except Exception:
        return []


async def get_authors_by_paper(paper_title: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper {title: $1})
        RETURN a.name, a.author_id
        """
    try:
        return await _run_query(query, [paper_title])
    except Exception:
        return []


async def get_paper_by_year(year: int) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {year: $1})
        RETURN p.title, p.doi, p.paper_id
        """
    try:
        return await _run_query(query, [year])
    except Exception:
        return []


async def get_papers_by_year_range(start_year: int, end_year: int) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE p.year >= $1 AND p.year <= $2
        RETURN p.title, p.year, p.doi, p.paper_id
        ORDER BY p.year DESC
        """
    try:
        return await _run_query(query, [start_year, end_year])
    except Exception:
        return []


async def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {paper_id: $1})
        RETURN p.title, p.year, p.doi, p.paper_id
        """
    try:
        rows = await _run_query(query, [paper_id])
        return rows[0] if rows else None
    except Exception:
        return None


async def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {doi: $1})
        RETURN p.title, p.year, p.paper_id
        """
    try:
        rows = await _run_query(query, [doi])
        return rows[0] if rows else None
    except Exception:
        return None


async def get_author_by_id(author_id: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (a:Author {author_id: $1})
        RETURN a.name, a.author_id
        """
    try:
        rows = await _run_query(query, [author_id])
        return rows[0] if rows else None
    except Exception:
        return None


async def search_papers_by_title(title_substring: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($1)
        RETURN p.title, p.year, p.doi, p.paper_id
        """
    try:
        return await _run_query(query, [title_substring])
    except Exception:
        return []


async def search_authors_by_name(name_substring: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)
        WHERE toLower(a.name) CONTAINS toLower($1)
        RETURN a.name, a.author_id
        """
    try:
        return await _run_query(query, [name_substring])
    except Exception:
        return []


async def get_all_papers(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = "MATCH (p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    params: List[Any] = []
    if limit:
        query += " LIMIT $1"
        params = [limit]
    try:
        return await _run_query(query, params)
    except Exception:
        return []


async def get_all_authors(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author) RETURN a.name, a.author_id ORDER BY a.name"
    params: List[Any] = []
    if limit:
        query += " LIMIT $1"
        params = [limit]
    try:
        return await _run_query(query, params)
    except Exception:
        return []


async def regex_for_paper(regex_string: str, limit: int = 100) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE p.title =~ $1
        RETURN p.title
        LIMIT $2
        """
    try:
        return await _run_query(query, [regex_string, limit])
    except Exception:
        return []


async def regex_for_author(regex_string: str, limit: int = 100) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)
        WHERE a.name =~ $1
        RETURN a.name
        LIMIT $2
        """
    try:
        return await _run_query(query, [regex_string, limit])
    except Exception:
        return []


# -- Update functions --
async def update_author(author_id: str, new_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not new_name:
        return None
    query = """
        MATCH (a:Author {author_id: $1})
        SET a.name = $2
        RETURN a.name, a.author_id
        """
    try:
        rows = await _run_query(query, [author_id, new_name])
        return rows[0] if rows else None
    except Exception:
        return None


async def update_paper(paper_id: str, title: Optional[str] = None, year: Optional[int] = None, doi: Optional[str] = None) -> Optional[Dict[str, Any]]:
    set_clauses: List[str] = []
    params: List[Any] = [paper_id]
    param_count = 2

    if title:
        set_clauses.append(f"p.title = ${param_count}")
        params.append(title)
        param_count += 1
    if year:
        set_clauses.append(f"p.year = ${param_count}")
        params.append(year)
        param_count += 1
    if doi:
        set_clauses.append(f"p.doi = ${param_count}")
        params.append(doi)
        param_count += 1

    if not set_clauses:
        return None

    query = f"""
        MATCH (p:Paper {{paper_id: $1}})
        SET {', '.join(set_clauses)}
        RETURN p.title, p.year, p.doi, p.paper_id
    """
    try:
        rows = await _run_query(query, params)
        return rows[0] if rows else None
    except Exception:
        return None


# -- Delete functions --
async def delete_author(author_id: str) -> bool:
    query = """
        MATCH (a:Author {author_id: $1})
        DETACH DELETE a
        """

    def _exec():
        try:
            _conn.execute(query, [author_id])
            return True
        except Exception as e:
            logging.error(f"Delete author failed: {e}")
            return False

    return await asyncio.to_thread(_exec)


async def delete_paper(paper_id: str) -> bool:
    query = """
        MATCH (p:Paper {paper_id: $1})
        DETACH DELETE p
        """

    def _exec():
        try:
            _conn.execute(query, [paper_id])
            return True
        except Exception as e:
            logging.error(f"Delete paper failed: {e}")
            return False

    return await asyncio.to_thread(_exec)


async def delete_authorship(author_id: str, paper_id: str) -> bool:
    query = """
        MATCH (a:Author {author_id: $1})-[r:WROTE]->(p:Paper {paper_id: $2})
        DELETE r
        """

    def _exec():
        try:
            _conn.execute(query, [author_id, paper_id])
            return True
        except Exception as e:
            logging.error(f"Delete authorship failed: {e}")
            return False

    return await asyncio.to_thread(_exec)


# -- Analytics / citation functions --
async def get_author_collaboration_count(author_id: str) -> int:
    query = """
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        RETURN count(DISTINCT a2) as collaborator_count
        """
    try:
        rows = await _run_query(query, [author_id])
        return rows[0]['collaborator_count'] if rows else 0
    except Exception:
        return 0


async def get_most_prolific_authors(limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        RETURN a.name, a.author_id, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT $1
        """
    try:
        return await _run_query(query, [limit])
    except Exception:
        return []


async def get_papers_per_year() -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        RETURN p.year, count(p) as paper_count
        ORDER BY p.year DESC
        """
    try:
        return await _run_query(query)
    except Exception:
        return []


async def get_citations_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1})
        RETURN citing.title, citing.year, citing.doi, citing.paper_id
        ORDER BY citing.year DESC
        """
    try:
        return await _run_query(query, [paper_id])
    except Exception:
        return []


async def get_references_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper {paper_id: $1})-[:CITES]->(cited:Paper)
        RETURN cited.title, cited.year, cited.doi, cited.paper_id
        ORDER BY cited.year DESC
        """
    try:
        return await _run_query(query, [paper_id])
    except Exception:
        return []


async def get_citation_count(paper_id: str) -> int:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1})
        RETURN count(citing) as citation_count
        """
    try:
        rows = await _run_query(query, [paper_id])
        return rows[0]['citation_count'] if rows else 0
    except Exception:
        return 0


async def get_most_cited_papers(limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper)
        RETURN cited.title, cited.year, cited.doi, cited.paper_id, count(citing) as citation_count
        ORDER BY citation_count DESC
        LIMIT $1
        """
    try:
        return await _run_query(query, [limit])
    except Exception:
        return []


async def get_citation_depth(paper_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
    query = """
        MATCH path = (start:Paper {paper_id: $1})-[:CITES*1..$2]->(end:Paper)
        RETURN end.title, end.year, end.paper_id, length(path) as depth
        ORDER BY depth, end.year DESC
        """
    try:
        return await _run_query(query, [paper_id, max_depth])
    except Exception:
        return []


async def get_co_citation_papers(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (p1:Paper {paper_id: $1})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
        WHERE p1 <> p2
        RETURN p2.title, p2.year, p2.paper_id, count(citing) as co_citation_count
        ORDER BY co_citation_count DESC
        LIMIT $2
        """
    try:
        return await _run_query(query, [paper_id, limit])
    except Exception:
        return []


async def detect_research_communities(min_cluster_size: int = 5) -> List[Dict[str, Any]]:
    query = """
        MATCH (a1:Author)-[:WROTE]->(p1:Paper)-[:CITES]->(p2:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        WITH a1, a2, count(*) as connection_strength
        WHERE connection_strength >= 2
        RETURN a1.name, a2.name, connection_strength
        ORDER BY connection_strength DESC
        """
    try:
        return await _run_query(query)
    except Exception:
        return []


async def track_keyword_citation_impact(keyword: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS $1 
              OR (p.abstract IS NOT NULL AND toLower(p.abstract) CONTAINS $1)

        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        WITH p, count(citing) as citation_count

        RETURN p.year, 
               count(p) as papers_with_keyword,
               avg(citation_count) as avg_citations_per_paper,
               sum(citation_count) as total_citations,
               max(citation_count) as max_citations
        ORDER BY p.year
        """
    try:
        return await _run_query(query, [keyword.lower()])
    except Exception:
        return []


async def track_keyword_temporal_trend(keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
    year_filter = ""
    params: List[Any] = [keyword.lower()]
    param_count = 2

    if start_year:
        year_filter += f"AND p.year >= ${param_count} "
        params.append(start_year)
        param_count += 1
    if end_year:
        year_filter += f"AND p.year <= ${param_count} "
        params.append(end_year)
        param_count += 1

    query = f"""
        MATCH (p:Paper)
        WHERE (toLower(p.title) CONTAINS $1 
               OR (p.keywords IS NOT NULL AND toLower(p.keywords) CONTAINS $1))
              {year_filter}
        RETURN p.year, count(p) as keyword_count
        ORDER BY p.year
        """
    try:
        return await _run_query(query, params)
    except Exception:
        return []


async def 
