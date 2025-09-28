i
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import threading
from contextlib import asynccontextmanager
import dspy

#define all the logging enums so that it is easier to call 

class MetricType(str , Enum): 
    QUERY_TIME = "query_time"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR_COUNT = "error_count"
    CONCURRENT_QUERIES = "concurrent_queries"
    DSPy_RESPONSE_TIME = "dsp_response_time"



