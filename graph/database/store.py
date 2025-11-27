import logging
import asyncio
import time
from typing import Any, List, Optional, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from neo4j import AsyncGraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "diam0ndman@3") 

@dataclass
class QueryResult:
    """Enhanced query result with metadata."""
    data: List[Dict[str, Any]]
    execution_time: float
    query: str
    params: List[Any]
    error: Optional[str] = None

class Neo4jConnectionPool:
    def __init__(self, uri: str, auth: tuple, pool_size: int = 10):
        self._driver = AsyncGraphDatabase.driver(
            uri, 
            auth=auth, 
            max_connection_pool_size=pool_size
        )
        self.pool_size = pool_size

    async def close(self):
        await self._driver.close()

    @asynccontextmanager
    async def get_connection(self):
        async with self._driver.session() as session:
            yield session

_connection_pool = Neo4jConnectionPool(URI, AUTH)

class EnhancedStore:
    def __init__(self, pool_size: int = 10, enable_cache: bool = True):
        self.pool = _connection_pool
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
    
    def _convert_params_to_dict(self, params: List[Any]) -> Dict[str, Any]:
        """Maps list params to $1, $2, etc. for Cypher compatibility."""
        if not params:
            return {}
        return {str(i+1): param for i, param in enumerate(params)}

    async def _execute_query(
        self, 
        query: str, 
        params: Optional[List[Any]] = None,
        cache_key: Optional[str] = None
    ) -> QueryResult:
        params = params or []
        
        # Check cache
        if self.enable_cache and cache_key and cache_key in self._cache:
            logging.info(f"Cache hit for key: {cache_key}")
            return self._cache[cache_key]
        
        start_time = time.time()
        mapped_params = self._convert_params_to_dict(params)
        data = []
        error = None

        try:
            async with self.pool.get_connection() as session:
                result = await session.run(query, mapped_params)
                data = await result.data()
        except Exception as e:
            logging.error(f"Query failed: {e}")
            error = str(e)
        
        result = QueryResult(
            data=data,
            execution_time=time.time() - start_time,
            query=query,
            params=params,
            error=error
        )
        
        if self.enable_cache and cache_key and not error:
            self._cache[cache_key] = result
        
        return result

    async def track_keyword_temporal_trend(self, keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
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
        result = await self._execute_query(query, params)
        return result.data

# Initialize Global Store Instance
_enhanced_store = EnhancedStore()

# --- STANDALONE WRAPPERS (For Action Dispatch) ---

async def _run_query(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    result = await _enhanced_store._execute_query(query, params)
    return result.data

# 1. Basic Getters
async def get_papers_by_author(author_name: str) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author {name: $1})-[:WROTE]->(p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    return await _run_query(query, [author_name])

async def get_authors_by_paper(paper_title: str) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author)-[:WROTE]->(p:Paper {title: $1}) RETURN a.name, a.author_id"
    return await _run_query(query, [paper_title])

async def get_authors_by_paper_id(paper_id: str) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author)-[:WROTE]->(p:Paper {paper_id: $1}) RETURN a.name, a.author_id"
    return await _run_query(query, [paper_id])

async def get_papers_by_author_id(author_id: str) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    return await _run_query(query, [author_id])

async def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    query = "MATCH (p:Paper {paper_id: $1}) RETURN p.title, p.year, p.doi, p.paper_id, p.publication_name, p.keywords"
    rows = await _run_query(query, [paper_id])
    return rows[0] if rows else None

async def get_author_by_id(author_id: str) -> Optional[Dict[str, Any]]:
    query = "MATCH (a:Author {author_id: $1}) RETURN a.name, a.author_id, a.affiliation"
    rows = await _run_query(query, [author_id])
    return rows[0] if rows else None

# 2. Advanced / Action Space Functions

async def get_citations_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    """Get papers that cite THIS paper (Incoming edges)."""
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1})
        RETURN citing.title, citing.year, citing.doi, citing.paper_id, citing.publication_name
        ORDER BY citing.year DESC
    """
    return await _run_query(query, [paper_id])

async def get_references_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    """Get papers THIS paper cites (Outgoing edges)."""
    query = """
        MATCH (source:Paper {paper_id: $1})-[:CITES]->(ref:Paper)
        RETURN ref.title, ref.year, ref.doi, ref.paper_id, ref.publication_name
        ORDER BY ref.year DESC
    """
    return await _run_query(query, [paper_id])

async def get_collabs_by_author(author_id: str) -> List[Dict[str , Any]]: 
    query = """ 
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        RETURN DISTINCT a2.name, a2.author_id, a2.affiliation
    """
    return await _run_query(query, [author_id])

async def get_papers_by_keyword(keyword: str, limit: int = 5, exclude_paper_id: str = None) -> List[Dict]: 
    query = """
        MATCH (p:Paper) 
        WHERE toLower(p.keywords) CONTAINS toLower($1)
    """
    params = [keyword]
    param_idx = 2

    if exclude_paper_id:
        query += f" AND p.paper_id <> ${param_idx} "
        params.append(exclude_paper_id)
        param_idx += 1
        
    query += f"""
        RETURN p.title, p.year, p.doi, p.paper_id, p.publication_name
        ORDER BY p.year DESC
        LIMIT ${param_idx}
    """
    params.append(limit)
    return await _run_query(query, params)

async def get_papers_by_venue(venue_name: str, exclude_paper_id: str = None, limit: int = 5) -> List[Dict]:
    # Maps 'venue' to 'publication_name'
    query = """
        MATCH (p:Paper)
        WHERE toLower(p.publication_name) CONTAINS toLower($1)
    """
    params = [venue_name]
    param_idx = 2
    
    if exclude_paper_id:
        query += f" AND p.paper_id <> ${param_idx} "
        params.append(exclude_paper_id)
        param_idx += 1
        
    query += f"""
        RETURN p.title, p.year, p.doi, p.keywords, p.paper_id
        ORDER BY p.year DESC 
        LIMIT ${param_idx}             
    """
    params.append(limit)
    return await _run_query(query, params)

async def get_older_references(paper_id: str) -> List[Dict]: 
    """Find references (outgoing) that are strictly older."""
    query = """
        MATCH (p:Paper {paper_id: $1})-[:CITES]->(ref:Paper)
        WHERE ref.year < p.year
        RETURN ref.doi, ref.title, ref.publication_name, ref.keywords, ref.paper_id
        ORDER BY ref.year ASC
    """
    return await _run_query(query, [paper_id])

async def get_newer_citations(paper_id: str) -> List[Dict]:
    """Find citations (incoming) that are strictly newer."""
    query = """
        MATCH (p:Paper {paper_id: $1})<-[:CITES]-(citing:Paper)
        WHERE citing.year > p.year 
        RETURN citing.doi, citing.title, citing.publication_name, citing.keywords, citing.paper_id
        ORDER BY citing.year DESC
    """
    return await _run_query(query, [paper_id])

async def get_second_degree_collaborators(author_id: str, limit: int = 20) -> List[Dict]:
    query = """
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(p1:Paper)<-[:WROTE]-(a2:Author)
        MATCH (a2)-[:WROTE]->(p2:Paper)<-[:WROTE]-(a3:Author)
        WHERE a1 <> a3 AND a1 <> a2
        RETURN DISTINCT a3.author_id, a3.name
        LIMIT $2
    """
    return await _run_query(query, [author_id, limit])

async def get_co_cited_neighbors(paper_id: str, limit: int = 10) -> List[Dict]:
    query = """
        MATCH (p1:Paper {paper_id: $1})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
        WHERE p1 <> p2
        RETURN p2.title, p2.year, p2.doi, p2.paper_id, count(citing) as co_citation_count
        ORDER BY co_citation_count DESC
        LIMIT $2
    """
    return await _run_query(query, [paper_id, limit])


async def count_prolific_publisher(paper_id: str , limit: int = 10) -> List[Dict]:
    query = """
            MATCH (p: Paper)
            WHERE p.publisher as publisher
            ORDER BY count(publisher) BY DESC 
            """
    
    return await _run_query(query , [paper_id , limit])


async def get_influence_path_papers(author_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Meta-Path Action: Finds papers cited by collaborators of the given author.
    (Author) -[WROTE]-> (P1) <-[WROTE]- (Collaborator) -[CITES]-> (P_Influence)
    """
    query = """
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
        WHERE a1 <> collab
        WITH collab
        MATCH (collab)-[:WROTE]->(p_collab:Paper)-[:CITES]->(p_influence:Paper)
        RETURN DISTINCT p_influence.title, p_influence.year, p_influence.doi, p_influence.paper_id
        ORDER BY p_influence.year DESC
        LIMIT $2
    """
    return await _run_query(query, [author_id, limit])


async def get_author_uni_collab_count(author_id: str) -> int:
    query = """
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2 AND a1.affiliation IS NOT NULL AND a1.affiliation = a2.affiliation
        RETURN count(DISTINCT a2) as uni_collaborator_count
    """
    rows = await _run_query(query, [author_id])
    return rows[0]['uni_collaborator_count'] if rows else 0

async def track_keyword_temporal_trend(keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
    return await _enhanced_store.track_keyword_temporal_trend(keyword, start_year, end_year)

async def get_all_papers(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = "MATCH (p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    params: List[Any] = []
    if limit:
        query += " LIMIT $1"
        params = [limit]
    return await _run_query(query, params)

