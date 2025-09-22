import asyncio
from store import *
from typing import List , Optional , Dict 



class AsyncFetch: 

    def __init__(self , max_workers = 4):
        self.max_workers = max_workers
    

    
    async def get_papers_by_author()