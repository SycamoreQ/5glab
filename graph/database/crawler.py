import requests
import os
from neo4j import GraphDatabase
import logging

API_KEY = "YOUR_ELSEVIER_API_KEY"
BASE_URL = "https://api.elsevier.com/content/search/scopus"
HEADERS = {"Accept": "application/json", "X-ELS-APIKey": API_KEY}


DB_URI = "bolt://localhost:7687" 
DB_AUTH = ("", "")

driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)

def setup_schema():
    """Create constraints to mimic Kuzu's PRIMARY KEY schema."""
    queries = [
        "CREATE CONSTRAINT ON (a:Author) ASSERT a.author_id IS UNIQUE;",
        "CREATE CONSTRAINT ON (p:Paper) ASSERT p.paper_id IS UNIQUE;",
        "CREATE INDEX ON :Paper(year);",
        "CREATE INDEX ON :Paper(title);"
    ]
    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except Exception as e:
                pass
    print("Schema constraints ensured.")

def fetch_papers(subject, count=10):
    url = f"{BASE_URL}?query=SUBJAREA({subject})&count={count}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("search-results", {}).get("entry", [])

def fetch_citations(scopus_id):
    """Fetch citations for a given paper using Elsevier API."""
    url = f"https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}"
    headers = {"Accept": "application/json", "X-ELS-APIKey": API_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        cited_refs = []
        ref_list = data.get("abstracts-retrieval-response", {}).get("references", {}).get("reference", [])
        if not isinstance(ref_list, list):
            ref_list = [ref_list]
            
        for ref in ref_list:
            ref_id = ref.get("scopus-id")
            if ref_id:
                cited_refs.append(ref_id)
        return cited_refs
    except Exception as e:
        logging.warning(f"Failed to fetch citations for {scopus_id}: {e}")
        return []

def db_is_empty():
    """Check if Paper nodes exist."""
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) as c")
        count = result.single()["c"]
        return count == 0

def clean_entry(e):
    """Helper to clean Scopus JSON into a dict."""
    return {
        "paper_id": e.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
        "title": e.get("dc:title", "").replace('"', "'"),
        "doi": e.get("prism:doi", ""),
        "publication_name": e.get("prism:publicationName", "").replace('"', "'"),
        "year": int(e.get("prism:coverDate", "").split("-")[0]) if e.get("prism:coverDate") else None,
        "keywords": e.get("authkeywords", "")
    }

def bulk_load(entries):
    """
    Load data using UNWIND (Batch Processing).
    This is the Memgraph/Neo4j equivalent of COPY FROM CSV for client-side scripts.
    """
    setup_schema()
    
    # Pre-process data into lists of dicts
    papers_data = []
    authors_data = []
    wrote_relations = []
    
    print(f"Preparing bulk load for {len(entries)} entries...")

    for e in entries:
        # Prepare Paper
        p_data = clean_entry(e)
        papers_data.append(p_data)
        
        # Prepare Authors and Relationships
        authors = e.get("author", [])
        if not isinstance(authors, list):
            authors = [authors]
            
        for a in authors:
            auth_id = a.get("authid")
            auth_name = a.get("authname", "")
            if auth_id:
                authors_data.append({"author_id": auth_id, "name": auth_name})
                wrote_relations.append({"author_id": auth_id, "paper_id": p_data["paper_id"]})

    with driver.session() as session:
        
        print("Inserting Papers...")
        session.run("""
            UNWIND $batch AS row
            MERGE (p:Paper {paper_id: row.paper_id})
            SET p += row
        """, batch=papers_data)

        print("Inserting Authors...")
        session.run("""
            UNWIND $batch AS row
            MERGE (a:Author {author_id: row.author_id})
            SET a.name = row.name
        """, batch=authors_data)
        
        print("Inserting Relationships...")
        session.run("""
            UNWIND $batch AS row
            MATCH (a:Author {author_id: row.author_id})
            MATCH (p:Paper {paper_id: row.paper_id})
            MERGE (a)-[:WROTE]->(p)
        """, batch=wrote_relations)

    print("Bulk load completed")

def insert_incremental(entries):
    """Insert new papers/authors transactionally."""
    # In Memgraph/Neo4j, we can actually reuse the bulk logic 
    # because UNWIND works efficiently for small batches too.
    # However, to keep your logic flow, we'll process them here.
    
    if not entries:
        return

    bulk_load(entries)
    print(f"Incrementally loaded {len(entries)} entries")

# Example Usage logic (similar to your main block)
if __name__ == "__main__":
    if db_is_empty():
        print("Database empty. Fetching initial batch...")
        data = fetch_papers("COMP", count=20)
        bulk_load(data)
    else:
        print("Database has data. Fetching incremental...")
        data = fetch_papers("COMP", count=5) # Example
        insert_incremental(data)