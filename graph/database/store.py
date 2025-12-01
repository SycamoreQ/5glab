import logging
import asyncio
import time
from typing import Any, List, Optional, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from neo4j import AsyncGraphDatabase
import threading

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

class MemgraphConnectionPool:
    """
    Wrapper around Neo4j/Memgraph Driver to maintain your existing API interface.
    The Neo4j driver manages its own pool internally, so we just wrap it.
    """
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
        """
        Yields a session. 
        """
        async with self._driver.session() as session:
            yield session

# Global connection pool
_connection_pool = MemgraphConnectionPool(URI, AUTH)

class EnhancedStore:
    """Store with async patterns and caching, adapted for Memgraph."""
    
    def __init__(self, pool_size: int = 10, enable_cache: bool = True):
        self.pool = _connection_pool # Use the global pool wrapper
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        # Executor not strictly needed for Neo4j async driver, but kept for structure
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
    
    def _convert_params_to_dict(self, params: List[Any]) -> Dict[str, Any]:
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
        
        # Check cache first
        if self.enable_cache and cache_key and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            logging.info(f"Cache hit for key: {cache_key}")
            return cached_result
        
        start_time = time.time()
        mapped_params = self._convert_params_to_dict(params)
        
        data = []
        error = None

        try:
            async with self.pool.get_connection() as session:
                # Run query
                result = await session.run(query, mapped_params)
                # Fetch records
                records = await result.data()
                data = records # result.data() returns List[Dict]
        except Exception as e:
            logging.error(f"Query failed: {e}")
            error = str(e)
        
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
        # Since the driver is async, we can map these to asyncio tasks directly
        tasks = []
        for query, params in queries:
            task = self._execute_query(query, params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
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
        queries = [
            ("MATCH (p:Paper {paper_id: $1}) RETURN p.title, p.year, p.doi, p.paper_id", [paper_id]),
            ("MATCH (a:Author)-[:WROTE]->(p:Paper {paper_id: $1}) RETURN a.name, a.author_id", [paper_id]),
            ("MATCH (citing:Paper)-[:CITES]->(cited:Paper {paper_id: $1}) RETURN citing.title, citing.year, citing.paper_id", [paper_id]),
            ("MATCH (citing:Paper {paper_id: $1})-[:CITES]->(cited:Paper) RETURN cited.title, cited.year, cited.paper_id", [paper_id]),
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
        queries = [
            ("MATCH (a:Author {author_id: $1}) RETURN a.name, a.author_id", [author_id]),
            ("MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper) RETURN p.title, p.year, p.paper_id ORDER BY p.year DESC", [author_id]),
            ("MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author) WHERE a1 <> a2 RETURN count(DISTINCT a2) as collaborator_count", [author_id]),
            ("MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:CITES]-(citing:Paper) RETURN count(citing) as total_citations", [author_id]),
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
        
        conditions = []
        params = []
        param_count = 1
        
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
        
        query = """
        UNWIND $papers AS paper
        MERGE (p:Paper {paper_id: paper.paper_id})
        SET p.title = paper.title, p.year = paper.year, p.doi = paper.doi
        
        WITH p, paper
        UNWIND paper.authors AS auth
        MERGE (a:Author {author_id: auth.author_id})
        SET a.name = auth.name
        MERGE (a)-[:WROTE]->(p)
        """
        
        start_time = time.time()
        
        chunk_size = 100
        processed = 0
        
        try:
            async with self.pool.get_connection() as session:
                for i in range(0, len(papers_data), chunk_size):
                    chunk = papers_data[i:i + chunk_size]
                    await session.run(query, {"papers": chunk})
                    processed += len(chunk)
                    
            return {
                "papers_processed": processed,
                "queries_executed": 1, 
                "successful": processed,
                "failed": 0,
                "total_execution_time": time.time() - start_time
            }
        except Exception as e:
            logging.error(f"Bulk insert failed: {e}")
            return {
                "papers_processed": 0,
                "error": str(e)
            }
        

    async def get_all_papers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "MATCH (p:Paper) RETURN p.title, p.year, p.doi, p.paper_id ORDER BY p.year DESC"
        params: List[Any] = []
        if limit:
            query += " LIMIT $1"
            params = [limit]
        result = await self._execute_query(query, params)
        return result.data
    
    async def get_all_authors(self, limit: int = 10) -> List[Dict[str, Any]]:
        query = "MATCH (a:Author) RETURN a.name, a.author_id ORDER BY a.name"
        params = []
        if limit:
            query += " LIMIT $1"
            params = [limit]
        result = await self._execute_query(query, params)
        return result.data

        

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        if not self.enable_cache:
            return {"cache_enabled": False}
        return {
            "cache_enabled": True,
            "cache_size": len(self._cache),
            "cached_queries": list(self._cache.keys())
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "connection_pool_size": self.pool.pool_size,
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._cache) if self._cache else 0,
            "executor_max_workers": self.executor._max_workers
        }


# Backwards compatibility
_enhanced_store = EnhancedStore()

async def _run_query(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
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

async def delete_author(author_id: str) -> bool:
    query = """
        MATCH (a:Author {author_id: $1})
        DETACH DELETE a
    """
    res = await _run_query(query, [author_id])
    return True 

async def delete_paper(paper_id: str) -> bool:
    query = """
        MATCH (p:Paper {paper_id: $1})
        DETACH DELETE p
    """
    res = await _run_query(query, [paper_id])
    return True

async def delete_authorship(author_id: str, paper_id: str) -> bool:
    query = """
        MATCH (a:Author {author_id: $1})-[r:WROTE]->(p:Paper {paper_id: $2})
        DELETE r
    """
    res = await _run_query(query, [author_id, paper_id])
    return True

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
        MATCH path = (start:Paper {paper_id: $1})-[:CITES*1..3]->(end:Paper)
        RETURN end.title, end.year, end.paper_id, length(path) as depth
        ORDER BY depth, end.year DESC
    """
    return await _run_query(query, [paper_id])

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



async def get_author_h_index(author_id: str) -> int:
    query = """
        MATCH (a:Author {author_id: $1})-[:WROTE]->(p:Paper)
        OPTIONAL MATCH (p2:Paper)-[:CITES]->(p)
        WITH p, count(p2) as citation_count
        RETURN collect(citation_count) as citations
    """
    rows = await _run_query(query, [author_id])
    if not rows:
        return 0
    citations = sorted(rows[0]['citations'], reverse=True)
    h_index = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h_index = i + 1
        else:
            break
    return h_index


async def get_author_uni_collab_count(author_id: str) -> int:
    query = """
        IF (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE a1 <> a2 AND a1.university = a2.university
        RETURN count(DISTINCT a2) as uni_collaborator_count
        """
    rows = await _run_query(query, [author_id])
    return rows[0]['uni_collaborator_count'] if rows else 0
    

async def get_collabs_by_author(author_id: str) -> List[Dict[str , Any]]: 
    query = """ 
            MATCH (a1:Author {author_id: $1})-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
            WHERE  a1 <> a2
            RETURN a2.name, a2.author_id"""
    
    return await _run_query(query, [author_id])


async def get_papers_by_keyword(self , keyword: str , limit = 5 , exclude_paper_id = None ) ->  List[Dict]: 
    query = """
            MATCH (p: Paper) 
            WHERE toLower(p.keyword) CONTAINS toLower ($1)
            """

    params = [keyword]

    if exclude_paper_id:
        query += " AND p.paper_id <> $2 "
        params.append(exclude_paper_id)
    query += """
            RETURN p.title , p.year , p.doi , p.paper_id
            ORDER BY p.year DESC
            LIMIT ($3)
            """
    return await _run_query(query , params + [limit])

async def get_papers_by_venue(venue_name: str , exclude_paper_id = None , limit = 5) -> List[Dict]:
    query = """
            MATCH (p: Paper)
            WHERE toLower(p.venue) CONTAINS toLower ($1)
            """
    
    params = [venue_name]
    
    if exclude_paper_id:
        query += """
                RETURN p.title , p.year , p.doi , p.keyword , p.paper_id
                ORDER BY p.year DESC 
                LIMIT ($3)             
                """
        
    return await _run_query(query , params + [limit])

async def get_older_references(paper_id: str) -> List[Dict]: 
    query = """
            MATCH (p: Paper {paper_id: $1})-[:CITES]->(ref: Paper)
            WHERE ref.year < p.year
            RETURN ref.doi , ref.title , ref.publisher , ref.venue , ref.keyword
            """
    
    return await _run_query(query , [paper_id])


async def get_newer_citations(paper_id: str) -> List[Dict]:
    query = """
            MATCH (p: Paper{paper_id: $1})-[:CITES]->(citing:Paper)
            WHERE citing.year > p.year 
            RETURN citing.doi , citing.title , citing.publisher , citing.venue , citing.keyword
            """
    
    return await _run_query(query  , [paper_id])


async def get_papers_in_year_window(year: int , window:int = 2) -> List[Dict]:
    query = """
            MATCH (p: Paper)
            WHERE p.year >= $1 - $2 AND p.year <= $1 + $2 
            RETURN p.doi , p.title , p.publisher , p.venue , p.keyword  
            """
    
    return await _run_query(query , [year , window])



    


async def track_keyword_temporal_trend(keyword: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
    return await _enhanced_store.track_keyword_temporal_trend(keyword, start_year, end_year)