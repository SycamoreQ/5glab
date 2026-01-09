from neo4j import AsyncGraphDatabase
import asyncio
from typing import Dict, List, Optional


class RemoteStore: 
    def __init__(self,
                host: str = "192.168.1.10",
                port: int = 7687,
                user: str = "neo4j", 
                password: str = "diam0ndman@3",
                database: str = "researchdbv3",
                pool_size: int = 10): 
        
        self.uri = f"neo4j://{host}:{port}"
        self.auth = (user, password)
        self.database = database
        self.pool_size = pool_size
        
        self.driver = None
        self._connected = False
        
    async def connect(self):
        if self._connected:
            return
        
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=self.auth,
            max_connection_pool_size=self.pool_size,
            connection_timeout=30.0
        )
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1")
                await result.single()
            
            self._connected = True
            print(f"Connected to remote database: {self.uri}")
        
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self):
        if self.driver:
            await self.driver.close()
            self._connected = False
            print("Disconnected from database")

            


    