import redis
from redis.commands.search.field import TextField, NumericField
from typing import List, Optional , Dict
import logging
import os 
from graph.database.store import * 
import hashlib
import pickle
from preprocess import AsyncResponseGenerator

@dataclass
class CacheConfig:
    redis_url: str = "redis://localhost:6379"
    llm_cache_db: int = 0  # Database 0 for LLM cache
    query_cache_db: int = 1  # Database 1 for query cache
    default_ttl: int = 3600  
    llm_cache_ttl: int = 7200 
    query_cache_ttl: int = 1800  
    max_connections: int = 10


class RedisCache: 

    def __init__(self, redis_url: str, db: int, max_connections: int = 10):
        self.redis_url = redis_url
        self.db = db
        self.max_connections = max_connections
        self._pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=False  # We'll handle encoding ourselves
            )
            self._redis = redis.Redis(connection_pool=self._pool)
        return self._redis
    

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            redis_client = await self._get_redis()
            cached_data = await redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            logging.error(f"Cache get error for key {key}: {e}")
            return None
        

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            redis_client = await self._get_redis()
            serialized_value = pickle.dumps(value)
            if ttl:
                return await redis_client.setex(key, ttl, serialized_value)
            else:
                return await redis_client.set(key, serialized_value)
        except Exception as e:
            logging.error(f"Cache set error for key {key}: {e}")
            return False
        

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            redis_client = await self._get_redis()
            return bool(await redis_client.delete(key))
        except Exception as e:
            logging.error(f"Cache delete error for key {key}: {e}")
            return False


    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(f"{prefix}:*")
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception as e:
            logging.error(f"Cache clear prefix error for {prefix}: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
        if self._pool:
            await self._pool.aclose()
    


class LLMCache(RedisCache):
    """Cache for LLM predictions from DSPy modules."""
    

    def __init__(self, config: CacheConfig):
        super().__init__(config.redis_url, config.llm_cache_db, config.max_connections)
        self.default_ttl = config.llm_cache_ttl

    
    async def get_prediction(self, module_name: str, signature: str, **inputs) -> Optional[Any]:
        """Get cached prediction for DSPy module."""
        key = self._generate_key(f"llm:{module_name}:{signature}", **inputs)
        return await self.get(key)
    
    
    async def cache_prediction(self, module_name: str, signature: str, inputs: Dict, prediction: Any, ttl: Optional[int] = None) -> bool:
        """Cache DSPy module prediction."""
        key = self._generate_key(f"llm:{module_name}:{signature}", **inputs)
        ttl = ttl or self.default_ttl
        return await self.set(key, prediction, ttl)
    
    
    async def invalidate_module(self, module_name: str) -> int:
        """Invalidate all cached predictions for a module."""
        return await self.clear_prefix(f"llm:{module_name}")



class QueryCache(RedisCache):

    def __init__(self , config: CacheConfig):
        super().__init__(config.redis_url , config.query_cache_db , config.max_connections)
        self.default_ttl = config.query_cache_ttl

    async def get_query_result(self, query: str, params: List[Any] = None) -> Optional[Any]:
        """Get cached query result."""
        params = params or []
        key = self._generate_key("query", query=query, params=params)
        return await self.get(key)
    
    async def cache_query_result(self, query: str, params: List[Any], result: Any, ttl: Optional[int] = None) -> bool:
        """Cache database query result."""
        params = params or []
        key = self._generate_key("query", query=query, params=params)
        ttl = ttl or self.default_ttl
        return await self.set(key, result, ttl)
    
    async def invalidate_queries_containing(self, table_name: str) -> int:
        """Invalidate queries that might involve a specific table."""
        # This is a simple approach - in production you might want more sophisticated invalidation
        pattern = f"*{table_name.lower()}*"
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(f"query:*")
            deleted = 0
            for key in keys:
                if table_name.lower() in key.decode().lower():
                    await redis_client.delete(key)
                    deleted += 1
            return deleted
        except Exception as e:
            logging.error(f"Query invalidation error: {e}")
            return 0
        

class CacheManager:
    """Manages both LLM and Query caches."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.llm_cache = LLMCache(self.config)
        self.query_cache = QueryCache(self.config)
    
    async def close(self):
        """Close both cache connections."""
        await self.llm_cache.close()
        await self.query_cache.close()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of both caches."""
        try:
            llm_redis = await self.llm_cache._get_redis()
            query_redis = await self.query_cache._get_redis()
            
            llm_ok = await llm_redis.ping()
            query_ok = await query_redis.ping()
            
            return {
                "llm_cache": llm_ok,
                "query_cache": query_ok,
                "overall": llm_ok and query_ok
            }
        except Exception as e:
            logging.error(f"Cache health check failed: {e}")
            return {"llm_cache": False, "query_cache": False, "overall": False}



def cached_prediction(cache_manager: CacheManager, module_name: str, ttl: Optional[int] = None):
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Extract signature name from function
            signature = getattr(func, '__name__', 'unknown')
            
            # Try to get from cache first
            cached_result = await cache_manager.llm_cache.get_prediction(
                module_name, signature, **kwargs
            )
            
            if cached_result is not None:
                logging.info(f"Cache hit for {module_name}:{signature}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_manager.llm_cache.cache_prediction(
                module_name, signature, kwargs, result, ttl
            )
            logging.info(f"Cached result for {module_name}:{signature}")
            
            return result
        return wrapper
    return decorator






