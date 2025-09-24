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
    
    def __init__(self , store_module):
        self.store = store_module
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
    
    
    async def 






        
            

        