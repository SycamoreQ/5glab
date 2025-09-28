import dspy 
import json 
from typing import List , Optional , Literal , Dict 
import logging 
import os 
import kuzu 
from kuzu import Database , AsyncConnection
import asyncio 
from preprocess import * 
from graph.database.store import * 




class Pipeline:
    
    def __init__(self ):
        self.store = EnhancedStore
        self.query_classifier = QueryClassifier()
        self.context_retriever = ContextRetriever()
        self.response_generator = ResponseGenerator()


    async def process_query(self , user_query:str) -> List[str , Any]:
        

        classification = await self.query_classifier(user_query)
        
        try: 
            entities = json.loads(classification.extracted_entities)
        except json.JSONDecodeError:
            entities = {}


        db_result = await self._exexute_database_queries(
            classification.query_type,
            entities
        )


        context_result = await self.context_retriever.forward(
            user_query = user_query,
            db_results = json.dumps(db_result , default=str),
            query_type = classification.query_type
        )

        response = await self.response_generator.forward(
            user_query=user_query,
            context=context_result.context,
            insights=context_result.key_insights
        )
        

        return {
            "answer": response.answer,
            "follow_up_suggestions": response.follow_up_suggestions,
            "query_classification": {
                "type": classification.query_type,
                "reasoning": classification.reasoning,
                "entities": entities
            },
            "raw_data": db_result
        }
    


    
    async def _execute_database_queries(self , query_type: QueryType , entities: Dict[str , Any]) -> Dict[str , Any]:
        result = {"query_type": QueryType , "data": []}

        try:
            
            if query_type == QueryType.AUTHOR_PAPERS:
                author_name = entities.get("author_name" , "")
                papers = await self.store.get_papers_by_author(author_name= author_name)

                result["data"] = papers

            elif query_type == QueryType.PAPER_AUTHORS:
                paper_title = entities.get("paper_title" , "")
                authors = await self.store.get_authors_by_paper(paper_title = paper_title)

                result["data"] = authors 

            elif query_type == QueryType.CITATION_TRENDS: 
                paper_id = entities.get("paper_id" , "")

                if paper_id : 
                    
                    citation_tasks = [
                        self.store.get_citations_by_paper(paper_id),
                        self.store.get_references_by_paper(paper_id),
                        self.store.get_citation_count(paper_id),
                        self.store.get_co_citation_papers(paper_id)
                        
                    ]

                    citations , references , count , co_citations = await asyncio.gather(*citation_tasks)

                    result["data"] = {
                        "citations" : citations,
                        "references": references,
                        "count" : count,
                        "co_citations": co_citations
                    }


            elif query_type == QueryType.TEMPORAL_TRENDS:
                keyword = entities.get("keyword", "")
                start_year = entities.get("start_year")
                end_year = entities.get("end_year")
                
                if keyword:
                    trend_data = await self.store.track_keyword_temporal_trend(
                        keyword, start_year, end_year
                    )
                    result["data"] = trend_data
            


            elif query_type == QueryType.COLLABORATION_NETWORK:
                author_id = entities.get("author_id", "")
                if author_id:
                    collab_count = await self.store.get_author_collaboration_count(author_id)
                    result["data"] = {"collaboration_count": collab_count}
            

            elif query_type == QueryType.KEYWORD_ANALYSIS:
                keyword = entities.get("keyword", "")
                if keyword:
                    # Search both papers and authors concurrently
                    paper_task = self.store.search_papers_by_title(keyword)
                    author_task = self.store.search_authors_by_name(keyword)
                    
                    papers, authors = await asyncio.gather(paper_task, author_task)
                    result["data"] = {"papers": papers, "authors": authors}
            


            elif query_type == QueryType.GENERAL_SEARCH:
                # For general queries, try multiple search strategies
                search_term = entities.get("search_term", entities.get("keyword", ""))
                if search_term:
                    search_tasks = [
                        self.store.search_papers_by_title(search_term),
                        self.store.search_authors_by_name(search_term),
                    ]
                    
                    papers, authors = await asyncio.gather(*search_tasks)
                    result["data"] = {"papers": papers, "authors": authors}
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    


# Batch processing for multiple queries
class AsyncBatchProcessor:
    def __init__(self, rag_pipeline: Pipeline):
        self.pipeline = rag_pipeline
    
    async def process_batch(self, queries: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently with rate limiting."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(query: str):
            async with semaphore:
                return await self.pipeline.process_query(query)
        
        tasks = [process_single(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": queries[i],
                    "error": str(result),
                    "answer": "An error occurred processing this query."
                })
            else:
                result["original_query"] = queries[i]
                processed_results.append(result)
        
        return processed_results



# Connection pool
class AsyncConnectionPool:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = asyncio.Queue()
        self._initialized = False
    

    async def _initialize_pool(self):
        """Initialize the connection pool."""
        import kuzu
        
        db = kuzu.Database(self.db_path)
        for _ in range(self.pool_size):
            conn = kuzu.AsyncConnection(db)
            await self._pool.put(conn)
        
        self._initialized = True
    

    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self._initialize_pool()
        
        return await self._pool.get()
    

    async def return_connection(self, conn):
        """Return a connection to the pool."""
        await self._pool.put(conn)
    

    async def execute_with_pool(self, query: str, params: Optional[List[Any]] = None):
        """Execute a query using a pooled connection."""
        conn = await self.get_connection()
        try:
            def _exec():
                result = conn.execute(query, params or [])
                return [dict(row) for row in result]
            
            result = await asyncio.to_thread(_exec)
            return result
        finally:
            await self.return_connection(conn)


                






        
             