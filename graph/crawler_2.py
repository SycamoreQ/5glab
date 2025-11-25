import requests
import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
import json 

DB_URI = "neo4j://localhost:7687"
DB_AUTH = ("neo4j", "diam0ndman@3") 
driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)


def inject_kaggle(filepath):

    with open(filepath, 'r') as f:
        papers = json.loads(f) 
    id2doi = {}

    

    

