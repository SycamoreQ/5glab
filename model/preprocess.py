import dspy
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from pydantic import BaseModel


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



class EntityHandler(dspy.Signature):

    sample: Author = dspy.InputField(desc = "this is a simple author field")


    




    