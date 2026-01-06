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

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
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
        async with self._driver.session(database="neo4j") as session:
            yield session

class EnhancedStore:
    """
    Store with async patterns and caching. All graph query functions are now
    methods of this class.
    UPDATED: Now uses new bulk loader schema with paperId and authorId
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
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields,
               p.citationCount as citation_count
        ORDER BY p.year DESC
        """
        return await self._run_query_method(query, [author_name])

    async def get_authors_by_paper(self, paper_title: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        WHERE p.title = $1
        RETURN a.authorId as author_id, a.name as name
        """
        return await self._run_query_method(query, [paper_title])

    async def get_authors_by_paper_id(self, paper_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper {paperId: $1})
        RETURN a.authorId as author_id, a.name as name
        """
        return await self._run_query_method(query, [paper_id])

    async def get_papers_by_author_id(self, author_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a:Author {authorId: $1})-[:WROTE]->(p:Paper)
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields,
               p.citationCount as citation_count
        ORDER BY p.year DESC
        """
        return await self._run_query_method(query, [author_id])

    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (p:Paper {paperId: $1})
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields,
               p.citationCount as citation_count,
               p.referenceCount as reference_count
        """
        rows = await self._run_query_method(query, [paper_id])
        return rows[0] if rows else None

    async def get_author_by_id(self, author_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (a:Author {authorId: $1})
        RETURN a.authorId as author_id, a.name as name
        """
        rows = await self._run_query_method(query, [author_id])
        return rows[0] if rows else None

    async def get_paper_by_title(self, title: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($1)
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields,
               p.citationCount as citation_count
        LIMIT 10
        """
        return await self._run_query_method(query, [title])

    async def get_citations_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers that cite THIS paper (Incoming edges)."""
        query = """
        MATCH (citing:Paper)-[:CITES]->(p:Paper {paperId: $1})
        RETURN citing.paperId as paper_id,
               citing.title as title,
               citing.year as year,
               citing.abstract as abstract,
               citing.venue as venue,
               citing.citationCount as citation_count
        ORDER BY citing.year DESC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_citation_count(self, paper_id: str) -> int:
        citations = await self.get_citations_by_paper(paper_id)
        return len(citations)

    async def get_collab_count(self, author_id: str) -> int:
        collabs = await self.get_collabs_by_author(author_id)
        return len(collabs)

    async def get_references_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers THIS paper cites (Outgoing edges)."""
        query = """
        MATCH (p:Paper {paperId: $1})-[:CITES]->(ref:Paper)
        RETURN ref.paperId as paper_id,
               ref.title as title,
               ref.year as year,
               ref.abstract as abstract,
               ref.venue as venue,
               ref.citationCount as citation_count
        ORDER BY ref.year DESC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_collabs_by_author(self, author_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a1:Author {authorId: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        RETURN DISTINCT a2.authorId as author_id, a2.name as name
        """
        return await self._run_query_method(query, [author_id])

    async def get_papers_by_keyword(self, keyword: str, limit: int = 5, exclude_paper_id: str = None) -> List[Dict]:
        query = """
        MATCH (p:Paper)
        WHERE any(field IN p.fieldsOfStudy WHERE toLower(field) CONTAINS toLower($1))
        """
        params = [keyword]
        param_idx = 2
        
        if exclude_paper_id:
            query += f" AND p.paperId <> ${param_idx} "
            params.append(exclude_paper_id)
            param_idx += 1
        
        query += f"""
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields
        ORDER BY p.citationCount DESC
        LIMIT ${param_idx}
        """
        params.append(limit)
        
        return await self._run_query_method(query, params)

    async def get_papers_by_venue(self, venue_name: str, exclude_paper_id: str = None, limit: int = 5) -> List[Dict]:
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.venue) CONTAINS toLower($1)
        """
        params = [venue_name]
        param_idx = 2
        
        if exclude_paper_id:
            query += f" AND p.paperId <> ${param_idx} "
            params.append(exclude_paper_id)
            param_idx += 1
        
        query += f"""
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields
        ORDER BY p.year DESC
        LIMIT ${param_idx}
        """
        params.append(limit)
        
        return await self._run_query_method(query, params)

    async def get_older_references(self, paper_id: str) -> List[Dict]:
        """Find references (outgoing) that are strictly older."""
        query = """
        MATCH (p:Paper {paperId: $1})-[:CITES]->(ref:Paper)
        WHERE ref.year < p.year
        RETURN ref.paperId as paper_id,
               ref.title as title,
               ref.year as year,
               ref.venue as venue,
               ref.fieldsOfStudy as fields
        ORDER BY ref.year ASC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_newer_citations(self, paper_id: str) -> List[Dict]:
        """Find citations (incoming) that are strictly newer."""
        query = """
        MATCH (citing:Paper)-[:CITES]->(p:Paper {paperId: $1})
        WHERE citing.year > p.year
        RETURN citing.paperId as paper_id,
               citing.title as title,
               citing.year as year,
               citing.venue as venue,
               citing.fieldsOfStudy as fields
        ORDER BY citing.year DESC
        """
        return await self._run_query_method(query, [paper_id])

    async def get_second_degree_collaborators(self, author_id: str, limit: int = 20) -> List[Dict]:
        query = """
        MATCH (a1:Author {authorId: $1})-[:WROTE]->(p1:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        MATCH (a2)-[:WROTE]->(p2:Paper)<-[:WROTE]-(a3:Author)
        WHERE a1 <> a3 AND a2 <> a3
        RETURN DISTINCT a3.authorId as author_id, a3.name as name
        LIMIT $2
        """
        return await self._run_query_method(query, [author_id, limit])

    async def get_co_cited_neighbors(self, paper_id: str, limit: int = 20) -> List[Dict]:
        query = """
        MATCH (p1:Paper {paperId: $1})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
        WHERE p1 <> p2
        RETURN p2.paperId as paper_id,
               p2.title as title,
               p2.year as year,
               count(citing) as co_citation_count
        ORDER BY co_citation_count DESC
        LIMIT $2
        """
        return await self._run_query_method(query, [paper_id, limit])

    async def get_basic_neighbors(self, paper_id):
        query = """
        MATCH (p:Paper {paperId: $1})-[:CITES|WROTE]-(neighbor)
        RETURN DISTINCT CASE
            WHEN neighbor:Paper THEN neighbor.paperId
            WHEN neighbor:Author THEN neighbor.authorId
        END as neighbor_id
        LIMIT 20
        """
        return await self._run_query_method(query, [paper_id])
    
    async def get_influence_path_papers(self, author_id: str, limit: int = 10):
        query = """
        MATCH (a:Author {authorId: $1})-[:WROTE]->(p1:Paper)
        MATCH (p1)-[:CITES]->(p2:Paper)
        MATCH (p2)<-[:WROTE]-(a2:Author)
        WITH DISTINCT p2
        ORDER BY p2.citationCount DESC
        LIMIT $2
        RETURN 
            p2.paperId as paper_id,
            p2.title as title,
            p2.abstract as abstract,
            p2.year as year,
            p2.citationCount as citation_count
        """
        return await self._run_query_method(query, [author_id, limit])


    async def get_any_paper(self) -> Optional[Dict[str, Any]]:
        """Get any random paper from the database (useful for initialization)"""
        query = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL AND p.year IS NOT NULL
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields
        LIMIT 1
        """
        rows = await self._run_query_method(query, [])
        return rows[0] if rows else None

    async def get_well_connected_paper(self) -> Optional[Dict[str, Any]]:
        """Get a paper with good connectivity (has both citations and references)"""
        query = """
        MATCH (p:Paper)
        WHERE p.citationCount > 5 AND p.referenceCount > 5
        OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        WITH p, count(DISTINCT ref) as ref_count, count(DISTINCT citing) as cite_count
        WHERE ref_count > 0 AND cite_count > 0
        RETURN p.paperId as paper_id,
               p.title as title,
               p.year as year,
               p.abstract as abstract,
               p.venue as venue,
               p.fieldsOfStudy as fields,
               ref_count,
               cite_count
        ORDER BY (ref_count + cite_count) DESC
        LIMIT 1
        """
        rows = await self._run_query_method(query, [])
        return rows[0] if rows else None

    async def get_prolific_authors(self) -> Optional[List]:
        query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
        OPTIONAL MATCH (a)-[:WROTE]->(:Paper)<-[:WROTE]-(coauthor:Author)
        WHERE a <> coauthor
        WITH a,
             count(DISTINCT p) as paper_count,
             count(DISTINCT citing) as total_citations,
             count(DISTINCT coauthor) as collaborator_count
        WHERE paper_count >= 5
        RETURN a.name as author_name,
               a.authorId as author_id,
               paper_count,
               total_citations,
               collaborator_count,
               (paper_count + total_citations/10.0 + collaborator_count) as impact_score
        ORDER BY impact_score DESC
        LIMIT 100
        """
        return await self._run_query_method(query, [])

    async def get_author_h_index(self, author_id: str) -> int:
        query = """
        MATCH (a:Author {authorId: $1})-[:WROTE]->(p:Paper)
        OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
        WITH p, count(citing) as citation_count
        ORDER BY citation_count DESC
        WITH collect(citation_count) as citation_counts
        UNWIND range(0, size(citation_counts)-1) as idx
        WITH idx, citation_counts[idx] as citations
        WHERE citations >= idx + 1
        RETURN max(idx + 1) as h_index
        """
        rows = await self._run_query_method(query, [author_id])
        return rows[0].get('h_index', 0) if rows and rows[0].get('h_index') else 0

    async def get_citation_velocity(self, author_id) -> int:
        query = """
        MATCH (a:Author {authorId: $1})-[:WROTE]->(p:Paper)
        WHERE p.year >= 2022
        OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
        RETURN count(citing) as recent_citations
        """
        result = await self._run_query_method(query, [author_id])
        return result[0].get('recent_citations', 0) if result and result[0].get('recent_citations') else 0

    async def get_pub_diversity(self, author_id) -> int:
        query = """
        MATCH (a:Author {authorId: $1})-[:WROTE]->(p:Paper)
        WITH count(DISTINCT p.fieldsOfStudy) as subjects, count(DISTINCT p.venue) as venues
        RETURN (subjects + venues) as diversity_score
        """
        result = await self._run_query_method(query, [author_id])
        return result[0].get('diversity_score', 0) if result and result[0].get('diversity_score') else 0

    async def get_author_by_name(self, author_name: str) -> List[Dict[str, Any]]:
        """Find author by name (fuzzy match)."""
        query = """
        MATCH (a:Author)
        WHERE toLower(a.name) CONTAINS toLower($1)
        RETURN a.authorId as author_id,
               a.name as name
        LIMIT 5
        """
        return await self._run_query_method(query, [author_name])

    async def get_collab_count_by_author(self, author_id: str) -> int:
        """Get number of unique collaborators for an author."""
        query = """
        MATCH (a:Author {authorId: $author_id})-[:WROTE]->(p:Paper)<-[:WROTE]-(collab:Author)
        WHERE collab.authorId <> $author_id
        RETURN COUNT(DISTINCT collab) as collab_count
        """
        result = await self._run_query_method(query, [author_id])
        return result[0].get('collab_count', 0) if result else 0

    async def get_author_by_name_fuzzy(self, name: str, limit: int = 5) -> List[Dict]:
        """Fuzzy search for authors by name."""
        query = """
        MATCH (a:Author)
        WHERE toLower(a.name) CONTAINS toLower($name)
        RETURN a
        ORDER BY a.paperCount DESC
        LIMIT $limit
        """
        results = await self._run_query_method(query, [name])
        return results[0].get('name' , '') if results else 0