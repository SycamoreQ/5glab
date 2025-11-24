import requests
import os
import time
import logging
from neo4j import GraphDatabase
from tqdm import tqdm # pip install tqdm

API_KEY = "YOUR_ELSEVIER_API_KEY"
BASE_URL = "https://api.elsevier.com/content/search/scopus"
HEADERS = {"Accept": "application/json", "X-ELS-APIKey": API_KEY}

DB_URI = "bolt://localhost:7687" 
DB_AUTH = ("", "")
driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)

def setup_schema():
    queries = [
        "CREATE CONSTRAINT ON (p:Paper) ASSERT p.paper_id IS UNIQUE;",
        "CREATE CONSTRAINT ON (a:Author) ASSERT a.author_id IS UNIQUE;",
        "CREATE INDEX ON :Paper(year);"
    ]
    with driver.session() as session:
        for q in queries:
            try: session.run(q)
            except: pass
    print("Schema constraints ensured.")

# --- 1. METADATA SEARCH (NODES) ---
def fetch_papers_metadata(subject, count=20):
    """Searches Scopus for papers to populate the graph nodes."""
    print(f"Searching for {count} papers on '{subject}'...")
    url = f"{BASE_URL}?query=SUBJAREA({subject})&count={count}"
    resp = requests.get(url, headers=HEADERS)
    return resp.json().get("search-results", {}).get("entry", [])

def clean_entry(e):
    return {
        "paper_id": e.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
        "title": e.get("dc:title", "").replace('"', "'"),
        "doi": e.get("prism:doi", ""),
        "publication_name": e.get("prism:publicationName", "").replace('"', "'"),
        "year": int(e.get("prism:coverDate", "").split("-")[0]) if e.get("prism:coverDate") else None,
        "keywords": e.get("authkeywords", "")
    }

def bulk_load_metadata(entries):
    setup_schema()
    papers_data = []
    authors_data = []
    wrote_relations = []

    for e in entries:
        p = clean_entry(e)
        if not p['paper_id']: continue
        papers_data.append(p)
        
        # Authors
        authors = e.get("author", [])
        if not isinstance(authors, list): authors = [authors]
        for a in authors:
            if a.get("authid"):
                authors_data.append({"author_id": a.get("authid"), "name": a.get("authname", "")})
                wrote_relations.append({"author_id": a.get("authid"), "paper_id": p["paper_id"]})

    with driver.session() as session:
        session.run("UNWIND $batch AS row MERGE (p:Paper {paper_id: row.paper_id}) SET p += row", batch=papers_data)
        session.run("UNWIND $batch AS row MERGE (a:Author {author_id: row.author_id}) SET a.name = row.name", batch=authors_data)
        session.run("UNWIND $batch AS row MATCH (a:Author {author_id: row.author_id}), (p:Paper {paper_id: row.paper_id}) MERGE (a)-[:WROTE]->(p)", batch=wrote_relations)
    
    print(f"Loaded {len(papers_data)} papers and metadata.")

def fetch_references(scopus_id):
    """
    Fetches the list of paper IDs that 'scopus_id' cites.
    REQUIRES: 'view=REF' in the API call.
    """
    url = f"https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}?view=REF"
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 429: # Rate limit
            time.sleep(1)
            return []
        if resp.status_code != 200:
            return []
            
        data = resp.json()
        refs = data.get("abstracts-retrieval-response", {}).get("references", {}).get("reference", [])
        
        # Scopus returns a list or a single dict if only 1 ref
        if isinstance(refs, dict): refs = [refs]
        
        cited_ids = []
        for r in refs:
            # We prefer Scopus ID, but can fallback to DOI if needed
            if r.get("scopus-id"):
                cited_ids.append(r.get("scopus-id"))
        return cited_ids
    except Exception as e:
        logging.error(f"Error fetching refs for {scopus_id}: {e}")
        return []

def enrich_graph_with_citations():
    """
    Iterates over papers in the DB that don't have CITATION data yet.
    Fetches their references and builds [:CITES] edges.
    """
    print("Starting Citation Graph Enrichment...")
    
    # 1. Find papers we haven't processed yet (Optimization)
    # We can flag them or just check for papers with 0 outgoing CITES edges
    # For now, we grab all papers in DB
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN p.paper_id AS pid")
        paper_ids = [record["pid"] for record in result]

    print(f"Found {len(paper_ids)} papers to check for citations.")
    
    citations_batch = []
    batch_size = 50

    for pid in tqdm(paper_ids):
        # Fetch who this paper cites
        cited_pids = fetch_references(pid)
        
        for cited_pid in cited_pids:
            citations_batch.append({"from": pid, "to": cited_pid})
            
        # Batch Insert
        if len(citations_batch) >= batch_size:
            _flush_citations(citations_batch)
            citations_batch = []
            
    # Flush remaining
    if citations_batch:
        _flush_citations(citations_batch)

def _flush_citations(batch):
    """
    Inserts CITES edges.
    CRITICAL: Creates 'Ghost Nodes' for papers we haven't scraped yet.
    """
    query = """
    UNWIND $batch AS row
    MATCH (source:Paper {paper_id: row.from})
    
    // MERGE the target (Ghost Node if it doesn't exist)
    MERGE (target:Paper {paper_id: row.to})
    
    // Create the edge
    MERGE (source)-[:CITES]->(target)
    """
    with driver.session() as session:
        session.run(query, batch=batch)

if __name__ == "__main__":
    # 1. Get Nodes
    if input("Fetch new papers? (y/n): ") == 'y':
        data = fetch_papers_metadata("COMP", count=10)
        bulk_load_metadata(data)
        
    # 2. Get Edges
    if input("Enrich citations? (y/n): ") == 'y':
        enrich_graph_with_citations()