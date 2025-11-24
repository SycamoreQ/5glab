import requests
import os
import logging
from neo4j import GraphDatabase
from utils.config import BaseConfig

API_KEY = BaseConfig.elsevier_api_key
BASE_URL = BaseConfig.elsevier_api_url
HEADERS = {
    "Accept": "application/json",
    "X-ELS-APIKey": API_KEY
}

DB_URI = "bolt://localhost:7687"
DB_AUTH = ("", "")
driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)

def setup_schema():
    """Create constraints and indexes for Authors and Papers."""
    queries = [
        "CREATE CONSTRAINT ON (a:Author) ASSERT a.author_id IS UNIQUE;",
        "CREATE CONSTRAINT ON (p:Paper) ASSERT p.paper_id IS UNIQUE;",
        "CREATE INDEX ON :Paper(year);",
        "CREATE INDEX ON :Paper(title);",
        "CREATE INDEX ON :Author(name);"
    ]
    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except Exception:
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
    try:
        response = requests.get(url, headers=HEADERS)
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
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) as c")
        return result.single()["c"] == 0

def clean_entry(e):
    """Clean Scopus JSON entry to internal paper dict."""
    return {
        "paper_id": e.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
        "title": e.get("dc:title", "").replace('"', "'"),
        "doi": e.get("prism:doi", ""),
        "publication_name": e.get("prism:publicationName", "").replace('"', "'"),
        "year": int(e.get("prism:coverDate", "").split("-")[0]) if e.get("prism:coverDate") else None,
        "keywords": e.get("authkeywords", "")
    }

def clean_author(a):
    """Extract all available author metadata."""
    return {
        "author_id": a.get("authid"),
        "name": a.get("authname", ""),
        "affiliation": a.get("affiliation", {}).get("affilname", "") if a.get("affiliation") else "",
        "orcid": a.get("orcid", ""),
        "seq": a.get("seq", ""),
        # Add more author fields as available/needed!
    }

def bulk_load(entries):
    setup_schema()
    papers_data = []
    authors_data = {}
    wrote_relations = []

    print(f"Preparing bulk load for {len(entries)} entries...")
    for e in entries:
        p_data = clean_entry(e)
        papers_data.append(p_data)

        authors = e.get("author", [])
        if not isinstance(authors, list):
            authors = [authors]
        for a in authors:
            auth_details = clean_author(a)
            # Use a dict to deduplicate by author_id
            if auth_details["author_id"]:
                authors_data[auth_details["author_id"]] = auth_details
                wrote_relations.append({
                    "author_id": auth_details["author_id"],
                    "paper_id": p_data["paper_id"]
                })

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
            SET a.name = row.name,
                a.affiliation = row.affiliation,
                a.orcid = row.orcid,
                a.seq = row.seq
        """, batch=list(authors_data.values()))

        print("Inserting Relationships...")
        session.run("""
            UNWIND $batch AS row
            MATCH (a:Author {author_id: row.author_id})
            MATCH (p:Paper {paper_id: row.paper_id})
            MERGE (a)-[:WROTE]->(p)
        """, batch=wrote_relations)
    print("Bulk load completed")

def insert_incremental(entries):
    if not entries:
        return
    bulk_load(entries)
    print(f"Incrementally loaded {len(entries)} entries")

