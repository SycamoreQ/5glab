import logging
import os
import asyncio
import pandas as pd
from neo4j import GraphDatabase

# MEMGRAPH CONFIGURATION
# Default Memgraph local settings
URI = "bolt://localhost:7687" 
AUTH = ("", "") # Memgraph usually has no auth by default, or ("username", "password")

# Setup synchronous driver for schema operations
driver = GraphDatabase.driver(URI, auth=AUTH)

def create_author_schema():
    """Create Author constraints (Memgraph equivalent of Node Table)"""
    try:
        with driver.session() as session:
            # Create constraint for primary key
            session.run("CREATE CONSTRAINT ON (a:Author) ASSERT a.author_id IS UNIQUE;")
            # Create index for name
            session.run("CREATE INDEX ON :Author(name);")
            
        print("Author schema/constraints created successfully")
        logging.info("Author schema created successfully")
    except Exception as e:
        logging.error(f"Schema creation failed: {e}")
        print(f"Author schema creation failed: {e}")

def create_paper_schema():
    """Create Paper constraints"""
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT ON (p:Paper) ASSERT p.paper_id IS UNIQUE;")
            session.run("CREATE INDEX ON :Paper(title);")
            session.run("CREATE INDEX ON :Paper(doi);")
            session.run("CREATE INDEX ON :Paper(year);")
            
        print("Paper schema/constraints created successfully")
        logging.info("Paper schema created successfully")
    except Exception as e:
        logging.error(f"Schema creation failed: {e}")
        print(f"Paper creation schema failed: {e}")

def wrote_relation():
    """Create WROTE relation - No explicit schema needed in Memgraph"""
    # Memgraph is schema-optional for edges, but we can verify connectivity
    print("Relation WROTE (schema-less) prepared")
    logging.info("relation WROTE prepared")

def cites_relation():
    """Create CITES relation"""
    print("Relation CITES (schema-less) prepared")
    logging.info("relation CITES prepared")

def similar_to_relation():
    """Create SIMILAR_TO relation"""
    print("Relation SIMILAR_TO (schema-less) prepared")
    logging.info("relation SIMILAR_TO prepared")

def drop_schema():
    """Drop all data - useful for testing/resetting"""
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Note: In Memgraph/Neo4j, constraints must be dropped separately if you want a full schema wipe,
            # but DETACH DELETE cleans all data.
        
        logging.info("Database data dropped successfully")
        print("Database data dropped successfully")
        return True
    except Exception as e:
        logging.error(f"Schema drop failed: {e}")
        print(f"Schema drop failed: {e}")
        return False

def load_csv_into_kuzu():
    """
    Load CSV files. 
    Note: Modified to read CSV in Python and push to Memgraph via Bolt.
    This ensures it works even if Memgraph is in Docker and can't see local files.
    """
    try:
        files_map = {
            'authors.csv': {'label': 'Author', 'query': "UNWIND $rows AS row MERGE (n:Author {author_id: row.author_id}) SET n += row"},
            'papers.csv':  {'label': 'Paper',  'query': "UNWIND $rows AS row MERGE (n:Paper {paper_id: row.paper_id}) SET n.title = row.title, n.doi = row.doi, n.year = toInteger(row.year), n.publication_name = row.publication_name"},
            'wrote.csv':   {'type': 'WROTE',   'query': "UNWIND $rows AS row MATCH (a:Author {author_id: row.from}), (p:Paper {paper_id: row.to}) MERGE (a)-[:WROTE]->(p)"},
            # 'cited.csv': {'type': 'CITES',   'query': "UNWIND $rows AS row MATCH (p1:Paper {paper_id: row.from}), (p2:Paper {paper_id: row.to}) MERGE (p1)-[:CITES]->(p2)"}
        }
        
        with driver.session() as session:
            for filename, config in files_map.items():
                if not os.path.exists(filename):
                    logging.warning(f"CSV file {filename} not found, skipping...")
                    continue
                
                print(f"Loading {filename}...")
                # Read CSV using pandas
                df = pd.read_csv(filename)
                
                # Replace NaN with None for Cypher compatibility
                data = df.where(pd.notnull(df), None).to_dict('records')
                
                # Batch process (Memgraph handles large batches well, but 1000-5000 is safe)
                batch_size = 2000
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    session.run(config['query'], rows=batch)
                    
        logging.info("CSV data loaded successfully")
        print("CSV data loaded successfully")
        return True
    
    except Exception as e:
        logging.error(f"CSV loading failed: {e}")
        print(f"CSV loading failed: {e}")
        return False

def index_tables():
    """Already handled in schema creation functions, keeping for compatibility"""
    return True