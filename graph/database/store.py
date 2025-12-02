import logging
import asyncio
import time
from typing import Any, List, Optional, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from neo4j import AsyncGraphDatabase
import os 
from typing import Tuple, Set

URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
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
    """
    Wrapper around the Neo4j Async Driver which manages its own connection pool.
    """
    def __init__(self, uri: str, auth: tuple, pool_size: int = 10):
        self._driver = AsyncGraphDatabase.driver(
            uri, 
            auth=auth, 
            max_connection_pool_size=pool_size,
            max_connection_lifetime=3600,  
            connection_acquisition_timeout=120.0
        )
        self.pool_size = pool_size

    async def close(self):
        await self._driver.close()

    @asynccontextmanager
    async def get_connection(self):
        """Yields an async session."""
        async with self._driver.session(database="researchdb") as session:
            yield session


class EnhancedStore:
    """
    Store with async patterns and caching. All graph query functions are now 
    methods of this class.
    
    FIXED: Now uses elementId() for node identification since paper_id is None in your DB
    """
    def __init__(self, pool_size: int = 10, enable_cache: bool = True):
        self.pool = Neo4jConnectionPool(URI, AUTH, pool_size)
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
        """Internal method to execute a raw Cypher query."""
        params = params or []
        
        # Check cache
        if self.enable_cache and cache_key and cache_key in self._cache:
            logging.debug(f"Cache hit for key: {cache_key}")
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

    async def _run_query_method(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Helper method that executes a query and returns the data list."""
        result = await self._execute_query(query, params)
        if result.error:
            logging.error(f"Database error during query: {result.error}")
            return []
        return result.data
        
    async def get_papers_by_author(self, author_name: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (a:Author {name: $1})-[:WROTE]->(p:Paper) 
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
            ORDER BY p.year DESC
        """
        return await self._run_query_method(query, [author_name])

    async def get_authors_by_paper(self, paper_title: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (a:Author)-[:WROTE]->(p:Paper {title: $1}) 
            RETURN elementId(a) as author_id, a.name, a.affiliation
        """
        return await self._run_query_method(query, [paper_title])

    async def get_authors_by_paper_id(self, paper_id: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (p:Paper), (a:Author)-[:WROTE]->(p)
            WHERE elementId(p) = $1
            RETURN elementId(a) as author_id, a.name, a.affiliation
        """
        return await self._run_query_method(query, [paper_id])

    async def get_papers_by_author_id(self, author_id: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (a:Author), (a)-[:WROTE]->(p:Paper)
            WHERE elementId(a) = $1
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
            ORDER BY p.year DESC
        """
        return await self._run_query_method(query, [author_id])

    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        query = """
            MATCH (p:Paper)
            WHERE elementId(p) = $1
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
        """
        rows = await self._run_query_method(query, [paper_id])
        return rows[0] if rows else None

    async def get_author_by_id(self, author_id: str) -> Optional[Dict[str, Any]]:
        query = """
            MATCH (a:Author)
            WHERE elementId(a) = $1
            RETURN elementId(a) as author_id, a.name, a.affiliation
        """
        rows = await self._run_query_method(query, [author_id])
        return rows[0] if rows else None

    async def get_paper_by_title(self, title: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (p:Paper) 
            WHERE p.title = $1 OR p.id = $1
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
        """
        rows = await self._run_query_method(query, [title])
        return rows

    async def get_citations_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers that cite THIS paper (Incoming edges)."""
        query = """
            MATCH (p:Paper), (citing:Paper)-[:CITES]->(p)
            WHERE elementId(p) = $1
            RETURN elementId(citing) as paper_id, 
                   COALESCE(citing.title, citing.id, '') as title,
                   citing.year, 
                   COALESCE(citing.doi, citing.id, '') as doi,
                   citing.publication_name, 
                   citing.keywords,
                   citing.id as original_id
            ORDER BY citing.year DESC
        """
        return await self._run_query_method(query, [paper_id])
    
    async def get_citation_count(self, paper_id: str) -> int: 
        citations = await self.get_citations_by_paper(paper_id)
        return len(citations)
    
    async def get_collab_count(self, author_id: str) -> int: 
        collab = await self.get_collabs_by_author(author_id)
        return len(collab)

    async def get_references_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers THIS paper cites (Outgoing edges)."""
        query = """
            MATCH (p:Paper), (p)-[:CITES]->(ref:Paper)
            WHERE elementId(p) = $1
            RETURN elementId(ref) as paper_id, 
                   COALESCE(ref.title, ref.id, '') as title,
                   ref.year, 
                   COALESCE(ref.doi, ref.id, '') as doi,
                   ref.publication_name, 
                   ref.keywords,
                   ref.id as original_id
            ORDER BY ref.year DESC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_collabs_by_author(self, author_id: str) -> List[Dict[str, Any]]: 
        query = """ 
            MATCH (a1:Author), (a1)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
            WHERE elementId(a1) = $1 AND a1 <> a2
            RETURN DISTINCT elementId(a2) as author_id, a2.name, a2.affiliation
        """
        return await self._run_query_method(query, [author_id])

    async def get_papers_by_keyword(self, keyword: str, limit: int = 5, exclude_paper_id: str = None) -> List[Dict]: 
        query = """
            MATCH (p:Paper) 
            WHERE toLower(p.keywords) CONTAINS toLower($1) OR toLower(p.id) CONTAINS toLower($1)
        """
        params = [keyword]
        param_idx = 2

        if exclude_paper_id:
            query += f" AND elementId(p) <> ${param_idx} "
            params.append(exclude_paper_id)
            param_idx += 1
            
        query += f"""
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
            ORDER BY p.year DESC
            LIMIT ${param_idx}
        """
        params.append(limit)
        return await self._run_query_method(query, params)

    async def get_papers_by_venue(self, venue_name: str, exclude_paper_id: str = None, limit: int = 5) -> List[Dict]:
        query = """
            MATCH (p:Paper)
            WHERE toLower(p.publication_name) CONTAINS toLower($1)
        """
        params = [venue_name]
        param_idx = 2
        
        if exclude_paper_id:
            query += f" AND elementId(p) <> ${param_idx} "
            params.append(exclude_paper_id)
            param_idx += 1
            
        query += f"""
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.keywords, 
                   p.publication_name,
                   p.id as original_id
            ORDER BY p.year DESC 
            LIMIT ${param_idx}             
        """
        params.append(limit)
        return await self._run_query_method(query, params)

    async def get_older_references(self, paper_id: str) -> List[Dict]: 
        """Find references (outgoing) that are strictly older."""
        query = """
            MATCH (p:Paper), (p)-[:CITES]->(ref:Paper)
            WHERE elementId(p) = $1 AND ref.year < p.year
            RETURN elementId(ref) as paper_id, 
                   COALESCE(ref.doi, ref.id, '') as doi,
                   COALESCE(ref.title, ref.id, '') as title,
                   ref.publication_name, 
                   ref.keywords,
                   ref.id as original_id
            ORDER BY ref.year ASC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_newer_citations(self, paper_id: str) -> List[Dict]:
        """Find citations (incoming) that are strictly newer."""
        query = """
            MATCH (p:Paper), (p)<-[:CITES]-(citing:Paper)
            WHERE elementId(p) = $1 AND citing.year > p.year 
            RETURN elementId(citing) as paper_id, 
                   COALESCE(citing.doi, citing.id, '') as doi,
                   COALESCE(citing.title, citing.id, '') as title,
                   citing.publication_name, 
                   citing.keywords,
                   citing.id as original_id
            ORDER BY citing.year DESC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_second_degree_collaborators(self, author_id: str, limit: int = 20) -> List[Dict]:
        query = """
            MATCH (a1:Author), (a1)-[:WROTE]->(p1:Paper)<-[:WROTE]-(a2:Author)
            WHERE elementId(a1) = $1
            MATCH (a2)-[:WROTE]->(p2:Paper)<-[:WROTE]-(a3:Author)
            WHERE a1 <> a3 AND a1 <> a2
            RETURN DISTINCT elementId(a3) as author_id, a3.name
            LIMIT $2
        """
        return await self._run_query_method(query, [author_id, limit])

    async def get_co_cited_neighbors(self, paper_id: str, limit: int = 20) -> List[Dict]:
        query = """
            MATCH (p1:Paper), (p1)<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
            WHERE elementId(p1) = $1 AND p1 <> p2
            RETURN elementId(p2) as paper_id, 
                   COALESCE(p2.title, p2.id, '') as title,
                   p2.year, 
                   COALESCE(p2.doi, p2.id, '') as doi,
                   count(citing) as co_citation_count,
                   p2.id as original_id
            ORDER BY co_citation_count DESC
            LIMIT $2
        """
        return await self._run_query_method(query, [paper_id, limit])
    

    async def get_basic_neighbors(self , paper_id):
        query = """
                MATCH (p:Paper)-[:CITES|WROTE]-(neighbor)
                WHERE elementId(p) = $1
                RETURN DISTINCT elementId(neighbor) as neighbor_id
                LIMIT 20
            """
        return await self._run_query_method(query, [paper_id])

    async def count_prolific_publisher(self, paper_id: str, limit: int = 10) -> List[Dict]:
        query = """
            MATCH (p:Paper)
            RETURN p.publisher AS publisher, count(p) AS count
            ORDER BY count DESC 
            LIMIT $1
        """
        return await self._run_query_method(query, [limit]) 

    async def get_influence_path_papers(self, author_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Meta-Path Action: Finds papers cited by collaborators of the given author.
        """
        query = """
            MATCH (a1:Author), (a1)-[:WROTE]->(:Paper)<-[:WROTE]-(collab:Author)
            WHERE elementId(a1) = $1 AND a1 <> collab
            WITH collab
            MATCH (collab)-[:WROTE]->(p_collab:Paper)-[:CITES]->(p_influence:Paper)
            RETURN DISTINCT elementId(p_influence) as paper_id, 
                   COALESCE(p_influence.title, p_influence.id, '') as title,
                   p_influence.year, 
                   COALESCE(p_influence.doi, p_influence.id, '') as doi,
                   p_influence.id as original_id
            ORDER BY p_influence.year DESC
            LIMIT $2
        """
        return await self._run_query_method(query, [author_id, limit])

    async def get_author_uni_collab_count(self, author_id: str) -> int:
        query = """
            MATCH (a1:Author), (a1)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
            WHERE elementId(a1) = $1 AND a1 <> a2 AND a1.affiliation IS NOT NULL AND a1.affiliation = a2.affiliation
            RETURN count(DISTINCT a2) as uni_collaborator_count
        """
        rows = await self._run_query_method(query, [author_id])
        return rows[0]['uni_collaborator_count'] if rows else 0
    
    async def get_any_paper(self) -> Optional[Dict[str, Any]]:
        """Get any random paper from the database (useful for initialization)"""
        query = """
            MATCH (p:Paper)
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id
            LIMIT 1
        """
        rows = await self._run_query_method(query, [])
        return rows[0] if rows else None
    
    async def get_well_connected_paper(self) -> Optional[Dict[str, Any]]:
        """Get a paper with good connectivity (has both citations and references)"""
        query = """
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            WITH p, count(DISTINCT ref) as ref_count, count(DISTINCT citing) as cite_count
            WHERE ref_count > 0 AND cite_count > 0
            RETURN elementId(p) as paper_id, 
                   COALESCE(p.title, p.id, '') as title,
                   p.year, 
                   COALESCE(p.doi, p.id, '') as doi,
                   p.publication_name, 
                   p.keywords,
                   p.id as original_id,
                   ref_count,
                   cite_count
            ORDER BY (ref_count + cite_count) DESC
            LIMIT 1
        """
        rows = await self._run_query_method(query, [])
        return rows[0] if rows else None
    

    async def get_well_connected_papers_from_list(self , cached_paper_ids) -> Optional[Dict[str , Any]]: 
        query = """
                MATCH (p: Paper)
                WHERE elementId(p) = $1 
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                WITH p, count(DISTINCT ref) as ref_count , count(DISTINCT citing) as cite_count 
                WHERE ref_count > 0 AND cite_count > 0
                RETURN elementId(p) as paper_id, 
                        COALESCE(p.title , p.id , '') as title,
                        p.year, 
                        COALESCE(p.doi , p.id , '') as doi,
                        p.publication_name,
                        p.keywords,
                        p.id as original_id,
                        ref_count,
                        cite_count,
                    ORDER BY (ref_count + cite_count) DESC 
                    LIMIT 1 
                """
        rows = await self._run_query_method(query, [])
        return rows[0] if rows else None
        

    