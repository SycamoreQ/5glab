import ray 
import time 
import heapq 
from typing import Dict , List , Optional 
from dataclasses import dataclass , field 
from core.job_spec import JobSpec , JobStatus , JobType
from graph.database.store import EnhancedStore
from RL.env import AdvancedGraphTraversalEnv


@dataclass 
class PrioritizeJob: 
    priority: int 
    timestamp: float = field(compare = True)
    query_density: float = field(compare= True)
    query_throughput: float = field(compare = True)


@dataclass

    