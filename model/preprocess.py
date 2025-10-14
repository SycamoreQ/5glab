import dspy
from typing import List, Optional, Literal , Dict
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from pydantic import BaseModel
import asyncio
from dspy import ChatAdapter 

#define the information required to extract from the graph db 
class Paper(BaseModel):
    paper_id: str
    title: str
    doi: Optional[str]
    publication_name: Optional[str]
    year: Optional[int]
    keywords: Optional[List[str]]

class Author(BaseModel):
    author_id: str 
    author_name: str


class QueryType(str , Enum):
    AUTHOR_PAPERS = "author_papers"
    PAPER_AUTHORS = "paper_authors"
    CITATION_TRENDS = "citation_trends"
    TEMPORAL_TRENDS = "temporal_trends"
    COLLABORATION_NETWORK = "collaboration_network"
    KEYWORD_ANALYSIS = "keyword_analysis"
    GENERAL_SEARCH = "general_search"



class SearchResult(str , Enum):
    paper:Optional[List[Paper]] = None 
    author:Optional[List[Author]] = None
    query: Optional[List[QueryType]] = None

    


class QueryClassifier: 
    """Classifies queries from user"""
    user_query: str = dspy.InputField(desc = "The users query about papers and authors in a research graph database")
    
    query_type: QueryType = dspy.OutputField(desc="The type of query being asked")
    extracted_entities: str = dspy.OutputField(desc="JSON string containing extracted entities like author names, paper titles, years, keywords")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this classification was chosen")



class ContextRetriever(dspy.Signature):
    """Generate contextual information from database results to help answer user queries."""
    
    user_query: str = dspy.InputField(desc="Original user query")
    db_results: str = dspy.InputField(desc="JSON formatted results from database queries")
    query_type: str = dspy.InputField(desc="Type of query that was executed")
    
    context: str = dspy.OutputField(desc="Relevant context extracted and formatted from the database results")
    key_insights: str = dspy.OutputField(desc="Important patterns or insights from the data")



class ResponseGenerator(dspy.Signature):
    """Generate a comprehensive answer to research queries using retrieved context."""
    
    user_query: str = dspy.InputField(desc="The original user question")
    context: str = dspy.InputField(desc="Relevant information retrieved from the database")
    insights: str = dspy.InputField(desc="Key insights from the data analysis")
    
    answer: str = dspy.OutputField(desc="Complete, well-structured answer to the user's query")
    follow_up_suggestions: str = dspy.OutputField(desc="Suggested follow-up questions or related queries")


query_classifier_template = ChatAdapter.from_messages([
    {"role": "system", "content": "You are a research query classifier that analyzes questions about authors, papers, and citations in a research knowledge graph."},
    {"role": "user", "content": (
        "Analyze the following query and classify it.\n\n"
        "Query: {user_query}\n\n"
        "You must output:\n"
        "- Type: one of [author_papers, paper_authors, citation_trends, temporal_trends, collaboration_network, keyword_analysis, general_search]\n"
        "- Entities: extract all relevant author names, paper titles, keywords, years, etc., as JSON\n"
        "- Reasoning: explain why this classification was chosen"
    )},
    {"role": "assistant", "content": (
        "Type: {query_type}\n"
        "Entities: {extracted_entities}\n"
        "Reasoning: {reasoning}"
    )}
])


context_retriever_template = ChatAdapter.from_messages([
    {"role": "system", "content": "You are a context retriever that summarizes database results to provide relevant information for answering research queries."},
    {"role": "user", "content": (
        "Given the user query and the following database results, extract and summarize the most relevant information.\n\n"
        "User Query: {user_query}\n\n"
        "Database Results: {db_results}\n\n"
        "Focus on information that directly addresses the query type: {query_type}.\n"
        "Provide a concise context summary and highlight key insights."
    )},
    {"role": "assistant", "content": (
        "Context: {context}\n"
        "Key Insights: {key_insights}"
    )}

])

response_generator_template = ChatAdapter.from_message([
    {"role": "system" , "content": "You are a response generator that crafts comprehensive answers to research queries using provided context and insights"},
    {"role": "user" , "content": (
        "Using the following context and insights, generate a clear and detailed answer to the user's query.\n\n"
        "User Query: {user_query}\n\n"
        "Context: {context}\n\n"
        "Key Insights: {insights}\n\n"
        "Your answer should be well-structured and informative. Also, suggest relevant follow-up questions or related topics the user might be interested in."
    )},
    {"role": "assistant" , "content": (
        "Answer: {answer}\n"
        "Follow-up Suggestions: {follow_up_suggestions}"
    )}
])





class QueryClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(QueryClassifier , template = query_classifier_template)
    
    async def forward(self, user_query: str):
        def _predict():
            return self.classifier(user_query=user_query)
        
        result = await asyncio.to_thread(_predict)
        return result
    



class AsyncContextRetriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = dspy.Predict(ContextRetriever , template = context_retriever_template)
    
    async def forward(self, user_query: str, db_results: str, query_type: str):
        def _predict():
            return self.retriever(
                user_query=user_query,
                db_results=db_results,
                query_type=query_type
            )
        
        result = await asyncio.to_thread(_predict)
        return result




class AsyncResponseGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(ResponseGenerator , template = response_generator_template)
    
    async def forward(self, user_query: str, context: str, insights: str):
        def _predict():
            return self.generator(
                user_query=user_query,
                context=context,
                insights=insights
            )
        
        result = await asyncio.to_thread(_predict)
        return result



    