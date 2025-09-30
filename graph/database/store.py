import kuzu
import logging
import asyncio
from kuzu import Database, Connection
from typing import Any, List, Optional, Dict, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import threading

DB_PATH = "research_db"

@dataclass
class QueryResult:
    """Enhanced query result with metadata."""
    data: List[Dict[str, Any]]
    execution_time: float
    query: str
    params: List[Any]
    error: Optional[str] = None

class ConnectionPool:
    """Thread-safe connection pool for Kuzu database."""
    
    def __init__(self, db_path: str, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self._db = Database(db_path)
        self._pool = []
        self._available = threading.Semaphore(pool_size)
        self._lock = threading.Lock()
        self._initialized = False
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                conn = kuzu.AsyncConnection(self._db)
                self._pool.append(conn)
            
            self._initialized = True
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool using context manager."""
        if not self._initialized:
            self._initialize_pool()
        
        # Wait for available connection
        await asyncio.to_thread(self._available.acquire)
        
        try:
            with self._lock:
                conn = self._pool.pop()
            yield conn
        finally:
            with self._lock:
                self._pool.append(conn)
            self._available.release()

# Global connection pool
_connection_pool = ConnectionPool(DB_PATH)




class EnhancedStore:
    """Store with async patterns and caching."""
    
    def __init__(self, pool_size: int = 10, enable_cache: bool = True):
        self.pool = ConnectionPool(DB_PATH, pool_size)
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
    
    async def _execute_query(
        self, 
        query: str, 
        params: Optional[List[Any]] = None,
        cache_key: Optional[str] = None
    ) -> QueryResult:
        """Enhanced query execution with timing and caching."""
        import time
        
        params = params or []
        
        # Check cache first
        if self.enable_cache and cache_key and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            logging.info(f"Cache hit for key: {cache_key}")
            return cached_result
        
        start_time = time.time()
        
        async with self.pool.get_connection() as conn:
            def _exec():
                try:
                    result = conn.execute(query, params)
                    return [dict(row) for row in result], None
                except Exception as e:
                    logging.error(f"Query failed: {e}")
                    return [], str(e)
            
            data, error = await asyncio.to_thread(_exec)
        
        execution_time = time.time() - start_time
        
        result = QueryResult(
            data=data,
            execution_time=execution_time,
            query=query,
            params=params,
            error=error
        )
        
        # Cache successful results
        if self.enable_cache and cache_key and not error:
            self._cache[cache_key] = result
        
        return result
    
    async def _execute_multiple_queries(
        self, 
        queries: List[tuple[str, List[Any]]]
    ) -> List[QueryResult]:
        """Execute multiple queries concurrently."""
        tasks = []
        for query, params in queries:
            task = self._execute_query(query, params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(QueryResult(
                    data=[],
                    execution_time=0.0,
                    query="",
                    params=[],
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    # Enhanced read functions with caching
    async def get_papers_by_author(self, author_name: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (a:Author {name: $1})-[:WROTE]->(p:Paper)
            RETURN p.title, p.year, p.doi, p.paper_id
            ORDER BY p.year DESC
        """
        cache_key = f"papers_by_author_{author_name}" if self.enable_cache else None
        result = await self._execute_query(query, [author_name], cache_key)
        return result.data
    
    async def get_authors_by_paper(self, paper_title: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (a:Author)-[:WROTE]->(p:Paper {title: $1})
            RETURN a.name, a.author_id
        """
        result = await self._execute_query(query, [paper_title])
        return result.data
    
    async def get_comprehensive_paper_info(self, paper_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a paper including authors and citations."""
        
        queries = [
            # Basic paper info
            ("MATCH (p:Paper {paper_id: $1}) RETURN p.title, p.year, p.doi, p.paper_id", [paper_id]),
            # Authors
            ("MATCH (a:Author)-[:WROTE]->(p:Paper {paper_id: $1}) RETURN a.name, a.author_id", [paper_id]),
            # Citations (papers citing this one)
            ("MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1}) RETURN citing.title, citing.year, citing.paper_id", [paper_id]),
            # References (papers this one cites)
            ("MATCH (citing:Paper {paper_id: $1})-[:CITES]->(cited:Paper) RETURN cited.title, cited.year, cited.paper_id", [paper_id]),
            # Citation count
            ("MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1}) RETURN count(citing) as citation_count", [paper_id])
        ]
        
        results = await self._execute_multiple_queries(queries)
        
        return {
            "paper_info": results[0].data[0] if results[0].data else None,
            "authors": results[1].data,
            "citations": results[2].data,
            "references": results[3].data,
            "citation_count": results[4].data[0]["citation_count"] if results[4].data else 0,
            "execution_times": [r.execution_time for r in results]
        }
    

    async def get_author_analytics(self, author_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for an author."""
        
        queries = [
            # Basic author info
            ("MATCH (a:Author {author_id: $1}) RETURN a.name, a.author_id", [author_id]),
            # Papers count and list
            ("MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper) RETURN p.title, p.year, p.paper_id ORDER BY p.year DESC", [author_id]),
            # Collaboration count
            ("MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author) WHERE a1 <> a2 RETURN count(DISTINCT a2) as collaborator_count", [author_id]),
            # Total citations received
            ("MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:CITES]-(citing:Paper) RETURN count(citing) as total_citations", [author_id]),
            # Publication years distribution
            ("MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper) RETURN p.year, count(p) as papers_count ORDER BY p.year", [author_id])
        ]
        
        results = await self._execute_multiple_queries(queries)
        
        return {
            "author_info": results[0].data[0] if results[0].data else None,
            "papers": results[1].data,
            "collaborator_count": results[2].data[0]["collaborator_count"] if results[2].data else 0,
            "total_citations": results[3].data[0]["total_citations"] if results[3].data else 0,
            "publication_timeline": results[4].data,
            "paper_count": len(results[1].data),
            "execution_times": [r.execution_time for r in results]
        }
    
    
    async def advanced_search(
        self, 
        title_keywords: Optional[List[str]] = None,
        author_keywords: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        min_citations: Optional[int] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Advanced search with multiple filters."""
        
        conditions = []
        params = []
        param_count = 1
        
        # Build dynamic query based on filters
        if title_keywords:
            title_conditions = []
            for keyword in title_keywords:
                title_conditions.append(f"toLower(p.title) CONTAINS toLower(${param_count})")
                params.append(keyword)
                param_count += 1
            conditions.append(f"({' OR '.join(title_conditions)})")
        
        if author_keywords:
            author_conditions = []
            for keyword in author_keywords:
                author_conditions.append(f"toLower(a.name) CONTAINS toLower(${param_count})")
                params.append(keyword)
                param_count += 1
            conditions.append(f"({' OR '.join(author_conditions)})")
        
        if year_range:
            conditions.append(f"p.year >= ${param_count} AND p.year <= ${param_count + 1}")
            params.extend(year_range)
            param_count += 2
        
        # Base query
        base_query = """
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        """
        
        if conditions:
            base_query += f" WHERE {' AND '.join(conditions)}"
        
        if min_citations:
            base_query += f" WITH p, a, count(citing) as citation_count WHERE citation_count >= ${param_count}"
            params.append(min_citations)
            param_count += 1
        else:
            base_query += " WITH p, a, count(citing) as citation_count"
        
        base_query += f"""
            RETURN p.title, p.year, p.paper_id, p.doi,
                   collect(DISTINCT a.name) as authors,
                   citation_count
            ORDER BY citation_count DESC, p.year DESC
            LIMIT ${param_count}
        """
        params.append(limit)
        
        result = await self._execute_query(base_query, params)
        
        return {
            "papers": result.data,
            "total_found": len(result.data),
            "execution_time": result.execution_time,
            "search_params": {
                "title_keywords": title_keywords,
                "author_keywords": author_keywords,
                "year_range": year_range,
                "min_citations": min_citations,
                "limit": limit
            }
        }
    
    async def get_research_trends(
        self, 
        keywords: List[str], 
        start_year: Optional[int] = None, 
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze research trends for multiple keywords over time."""
        
        tasks = []
        for keyword in keywords:
            task = self.track_keyword_temporal_trend(keyword, start_year, end_year)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        trend_data = {}
        for i, keyword in enumerate(keywords):
            trend_data[keyword] = results[i]
        
        return {
            "trends": trend_data,
            "keywords": keywords,
            "date_range": {"start_year": start_year, "end_year": end_year}
        }
    

    async def track_keyword_temporal_trend(self, keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enhanced version with caching."""
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
        
        cache_key = f"trend_{keyword}_{start_year}_{end_year}" if self.enable_cache else None
        result = await self._execute_query(query, params, cache_key)
        return result.data
    
    async def bulk_insert_papers(self, papers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk insert papers with batch processing."""
        
        # Prepare batch insert queries
        paper_queries = []
        author_queries = []
        relationship_queries = []
        
        for paper_data in papers_data:
            paper_queries.append((
                "CREATE (p:Paper {paper_id: $1, title: $2, year: $3, doi: $4})",
                [paper_data.get('paper_id'), paper_data.get('title'), 
                 paper_data.get('year'), paper_data.get('doi')]
            ))
            
            # Author insertions and relationships
            for author in paper_data.get('authors', []):
                author_queries.append((
                    "MERGE (a:Author {author_id: $1, name: $2})",
                    [author.get('author_id'), author.get('name')]
                ))
                
                relationship_queries.append((
                    "MATCH (a:Author {author_id: $1}), (p:Paper {paper_id: $2}) CREATE (a)-[:WROTE]->(p)",
                    [author.get('author_id'), paper_data.get('paper_id')]
                ))
        
        all_queries = paper_queries + author_queries + relationship_queries
        
        # Process in chunks
        chunk_size = 100
        results = []
        
        for i in range(0, len(all_queries), chunk_size):
            chunk = all_queries[i:i + chunk_size]
            chunk_results = await self._execute_multiple_queries(chunk)
            results.extend(chunk_results)
        
        successful = sum(1 for r in results if not r.error)
        failed = len(results) - successful
        
        return {
            "papers_processed": len(papers_data),
            "queries_executed": len(results),
            "successful": successful,
            "failed": failed,
            "total_execution_time": sum(r.execution_time for r in results)
        }
    
    def clear_cache(self):
        """Clear the query cache."""
        if self._cache:
            self._cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self._cache),
            "cached_queries": list(self._cache.keys())
        }
    
    # Performance monitoring
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "connection_pool_size": self.pool.pool_size,
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._cache) if self._cache else 0,
            "executor_max_workers": self.executor._max_workers
        }


# Backwards compatibility - maintain your existing interface
_enhanced_store = EnhancedStore()


async def _run_query(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Backwards compatible query runner."""
    result = await _enhanced_store._execute_query(query, params)
    return result.data




async def get_papers_by_author(author_name: str) -> List[Dict[str, Any]]:
    return await _enhanced_store.get_papers_by_author(author_name)

async def get_authors_by_paper(paper_title: str) -> List[Dict[str, Any]]:
    return await _enhanced_store.get_authors_by_paper(paper_title)

async def get_paper_by_year(year: int) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {year: $1})
        RETURN p.title, p.doi, p.paper_id
    """
    return await _run_query(query, [year])



async def get_papers_by_year_range(start_year: int, end_year: int) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE p.year >= $1 AND p.year <= $2
        RETURN p.title, p.year, p.doi, p.paper_id
        ORDER BY p.year DESC
    """
    return await _run_query(query, [start_year, end_year])



async def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {paper_id: $1})
        RETURN p.title, p.year, p.doi, p.paper_id
    """
    rows = await _run_query(query, [paper_id])
    return rows[0] if rows else None



async def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (p:Paper {doi: $1})
        RETURN p.title, p.year, p.paper_id
    """
    rows = await _run_query(query, [doi])
    return rows[0] if rows else None



async def get_author_by_id(author_id: str) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (a:Author {author_id: $1})
        RETURN a.name, a.author_id
    """
    rows = await _run_query(query, [author_id])
    return rows[0] if rows else None



async def search_papers_by_title(title_substring: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($1)
        RETURN p.title, p.year, p.doi, p.paper_id
    """
    return await _run_query(query, [title_substring])



async def search_authors_by_name(name_substring: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)
        WHERE toLower(a.name) CONTAINS toLower($1)
        RETURN a.name, a.author_id
    """
    return await _run_query(query, [name_substring])



async def get_all_papers(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = "MATCH (p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
    params: List[Any] = []
    if limit:
        query += " LIMIT $1"
        params = [limit]
    return await _run_query(query, params)



async def get_all_authors(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = "MATCH (a:Author) RETURN a.name, a.author_id ORDER BY a.name"
    params: List[Any] = []
    if limit:
        query += " LIMIT $1"
        params = [limit]
    return await _run_query(query, params)



async def regex_for_paper(regex_string: str, limit: int = 100) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        WHERE p.title =~ $1
        RETURN p.title
        LIMIT $2
    """
    return await _run_query(query, [regex_string, limit])



async def regex_for_author(regex_string: str, limit: int = 100) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)
        WHERE a.name =~ $1
        RETURN a.name
        LIMIT $2
    """
    return await _run_query(query, [regex_string, limit])



# Update functions
async def update_author(author_id: str, new_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not new_name:
        return None
    query = """
        MATCH (a:Author {author_id: $1})
        SET a.name = $2
        RETURN a.name, a.author_id
    """
    rows = await _run_query(query, [author_id, new_name])
    return rows[0] if rows else None



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
    rows = await _run_query(query, params)
    return rows[0] if rows else None



# Delete functions
async def delete_author(author_id: str) -> bool:
    query = """
        MATCH (a:Author {author_id: $1})
        DETACH DELETE a
    """
    
    async with _enhanced_store.pool.get_connection() as conn:
        def _exec():
            try:
                conn.execute(query, [author_id])
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
    
    async with _enhanced_store.pool.get_connection() as conn:
        def _exec():
            try:
                conn.execute(query, [paper_id])
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
    
    async with _enhanced_store.pool.get_connection() as conn:
        def _exec():
            try:
                conn.execute(query, [author_id, paper_id])
                return True
            except Exception as e:
                logging.error(f"Delete authorship failed: {e}")
                return False
        
        return await asyncio.to_thread(_exec)



# Analytics functions
async def get_author_collaboration_count(author_id: str) -> int:
    query = """
        MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        RETURN count(DISTINCT a2) as collaborator_count
    """
    rows = await _run_query(query, [author_id])
    return rows[0]['collaborator_count'] if rows else 0



async def get_most_prolific_authors(limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        RETURN a.name, a.author_id, count(p) as paper_count
        ORDER BY paper_count DESC
        LIMIT $1
    """
    return await _run_query(query, [limit])



async def get_papers_per_year() -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Paper)
        RETURN p.year, count(p) as paper_count
        ORDER BY p.year DESC
    """
    return await _run_query(query)



async def get_citations_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1})
        RETURN citing.title, citing.year, citing.doi, citing.paper_id
        ORDER BY citing.year DESC
    """
    return await _run_query(query, [paper_id])



async def get_references_by_paper(paper_id: str) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper {paper_id: $1})-[:CITES]->(cited:Paper)
        RETURN cited.title, cited.year, cited.doi, cited.paper_id
        ORDER BY cited.year DESC
    """
    return await _run_query(query, [paper_id])



async def get_citation_count(paper_id: str) -> int:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1})
        RETURN count(citing) as citation_count
    """
    rows = await _run_query(query, [paper_id])
    return rows[0]['citation_count'] if rows else 0



async def get_most_cited_papers(limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper)
        RETURN cited.title, cited.year, cited.doi, cited.paper_id, count(citing) as citation_count
        ORDER BY citation_count DESC
        LIMIT $1
    """
    return await _run_query(query, [limit])



async def get_citation_depth(paper_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
    query = """
        MATCH path = (start:Paper {paper_id: $1})-[:CITES*1..$2]->(end:Paper)
        RETURN end.title, end.year, end.paper_id, length(path) as depth
        ORDER BY depth, end.year DESC
    """
    return await _run_query(query, [paper_id, max_depth])



async def get_co_citation_papers(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        MATCH (p1:Paper {paper_id: $1})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
        WHERE p1 <> p2
        RETURN p2.title, p2.year, p2.paper_id, count(citing) as co_citation_count
        ORDER BY co_citation_count DESC
        LIMIT $2
    """
    return await _run_query(query, [paper_id, limit])



async def detect_research_communities(min_cluster_size: int = 5) -> List[Dict[str, Any]]:
    query = """
        MATCH (a1:Author)-[:WROTE]->(p1:Paper)-[:CITES]->(p2:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2
        WITH a1, a2, count(*) as connection_strength
        WHERE connection_strength >= 2
        RETURN a1.name, a2.name, connection_strength
        ORDER BY connection_strength DESC
    """
    return await _run_query(query)



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
    return await _run_query(query, [keyword.lower()])



async def track_keyword_temporal_trend(keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
    return await _enhanced_store.track_keyword_temporal_trend(keyword, start_year, end_year)