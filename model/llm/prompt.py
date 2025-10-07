#create/generate prompts using dspy prompt optimizers and then load into llm 

import dspy 
from dspy.teleprompt import MIPROv2
import asyncio  
from typing import List , Optional , Literal , Dict , Any 
from pipeline import Pipeline
from dataclasses import dataclass
import json
from graph.database.store import EnhancedStore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



#only for prompt optimization and not for complete trainset and valset creation 

@dataclass
class TrainingExample:
    """Training example for DSPy optimization."""
    query: str
    expected_query_type: str
    expected_entities: Dict[str, Any]
    db_results: str
    expected_answer: str
    
    def to_dspy_example(self):
        return dspy.Example(
            user_query=self.query,
            db_results=self.db_results,
            answer=self.expected_answer
        ).with_inputs('user_query', 'db_results')


def create_training_data_from_kuzu(kuzu_db_path: str) -> List[TrainingExample]:
    """
    Generate training examples from your Kuzu graph database.
    This creates realistic query-answer pairs for optimization.
    """
    from kuzu import Database, Connection
    
    db = Database(kuzu_db_path)
    conn = Connection(db)
    
    examples = []
    
    query = """
    MATCH (a:Author)-[:AUTHORED]->(p:Paper)
    RETURN a.name as author, p.title as title, p.year as year
    LIMIT 50
    """
    
    results = conn.execute(query)
    author_papers = {}
    
    while results.has_next():
        row = results.get_next()
        author = row[0]
        if author not in author_papers:
            author_papers[author] = []
        author_papers[author].append({'title': row[1], 'year': row[2]})
    
    for author, papers in list(author_papers.items())[:20]:
        db_result = json.dumps({
            "query_type": "author_papers",
            "data": papers
        })
        
        paper_list = ", ".join([f'"{p["title"]}" ({p["year"]})' for p in papers[:3]])
        
        example = TrainingExample(
            query=f"What papers has {author} written?",
            expected_query_type="author_papers",
            expected_entities={"author_name": author},
            db_results=db_result,
            expected_answer=f"{author} has authored {len(papers)} papers including {paper_list}."
        )
        examples.append(example)

    
    query = """
    MATCH (p1:Paper)-[:CITES]->(p2:Paper)
    RETURN p1.title as citing, p2.title as cited, p2.year as year
    LIMIT 50
    """
    
    results = conn.execute(query)
    citations = []
    
    while results.has_next():
        row = results.get_next()
        citations.append({
            'citing': row[0],
            'cited': row[1],
            'year': row[2]
        })
    
    for citation in citations[:20]:
        db_result = json.dumps({
            "query_type": "citation_trends",
            "data": {
                "citations": [citation],
                "count": 1
            }
        })
        
        example = TrainingExample(
            query=f"Which papers cite {citation['cited']}?",
            expected_query_type="citation_trends",
            expected_entities={"paper_title": citation['cited']},
            db_results=db_result,
            expected_answer=f'"{citation["citing"]}" cites "{citation["cited"]}" (published in {citation["year"]}).'
        )
        examples.append(example)

    similarity = []

    query = """
    MATCH (p1:Paper)-[:SIMILAR_TO]->(p2:Paper)
    RETURN p1.title as title_1 , p2.title as title_2
    LIMIT 10
    """

    while results.has_next():
        row = results.get_next()
        similarity.append({
            'title_1': row[0],
            'title_2': row[1],
        })
    
    for similarity in similarity[:20]:
        db_result = json.dumps({
            "query_type": "paper_similarity",
            "data": {
                "similar_papers": [similarity],
                "count": 1
            }
        })
        
        example = TrainingExample(
            query=f"Is the first paper {similarity[0]} similar to the second paper {similarity[1]}?",
            expected_query_type="paper_similarity",
            expected_entities={"title_1": similarity[0] , "title_2": similarity[1]},
            db_results=db_result,
            expected_answer=f'{similarity[0]} is similar with respect to content , author and other factors with {similarity[1]}.'
        )
        examples.append(example)
    
    return examples


def optimize_pipeline_with_mipro(
    pipeline: Pipeline,
    trainset: List[dspy.Example],
    valset: List[dspy.Example]
):
    """
    Use DSPy's MIPRO optimizer to find best prompts.
    This is CRITICAL before fine-tuning!
    """
    from dspy.teleprompt import MIPROv2
    
    def answer_quality_metric(example, pred, is_similarity: bool ,  trace=None):
        """Evaluate if answer is relevant and accurate."""
        expected_keywords = set(example.answer.lower().split())
        pred_keywords = set(pred.answer.lower().split())

        similarity = cosine_similarity(np.array(list(expected_keywords)).reshape(1, -1) , np.array(list(pred_keywords)).reshape(1, -1))
        return similarity[0][0] if similarity[0][0] > 0.5 else 0.0 # Return similarity score if above threshold, else 0
        
    optimizer = MIPROv2(
        metric=answer_quality_metric,
        num_candidates=10,
        init_temperature=1.0
    )
    
    print("Optimizing DSPy pipeline...")
    optimized_pipeline = optimizer.compile(
        pipeline,
        trainset=trainset,
        num_trials=20,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        eval_kwargs={'num_threads': 4}
    )
    
    return optimized_pipeline


