import csv
import requests
from kuzu import Database, Connection
import os 


API_KEY = "YOUR_ELSEVIER_API_KEY"
BASE_URL = "https://api.elsevier.com/content/search/scopus"
HEADERS = {"Accept": "application/json", "X-ELS-APIKey": API_KEY}

# KuzuDB
DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)


def fetch_papers(subject, count=10):
    url = f"{BASE_URL}?query=SUBJAREA({subject})&count={count}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("search-results", {}).get("entry", [])


def db_is_empty():
    """Check if Paper table has rows."""
    try:
        result = conn.execute("MATCH (p:Paper) RETURN COUNT(p) AS c;")
        count = result[0]["c"]
        return count == 0
    except Exception:
        return True 


def bulk_load(entries):
    """Dump JSON → CSV and use COPY FROM for first load."""
    os.makedirs("csv_data", exist_ok=True)
    authors_csv = "csv_data/authors.csv"
    papers_csv = "csv_data/papers.csv"
    wrote_csv = "csv_data/wrote.csv"

    with open(authors_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["author_id", "name"])
        for e in entries:
            authors = e.get("author", [])
            if not isinstance(authors, list):
                authors = [authors]
            for a in authors:
                if a.get("authid"):
                    writer.writerow([a.get("authid"), a.get("authname", "")])

    with open(papers_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "title", "doi", "publication_name", "year", "keywords"])
        for e in entries:
            pid = e.get("dc:identifier", "").replace("SCOPUS_ID:", "")
            title = e.get("dc:title", "").replace('"', "'")
            doi = e.get("prism:doi", "")
            pub = e.get("prism:publicationName", "").replace('"', "'")
            year = e.get("prism:coverDate", "").split("-")[0] if e.get("prism:coverDate") else ""
            keywords = e.get("authkeywords", "")
            writer.writerow([pid, title, doi, pub, year, keywords])

    with open(wrote_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["author_id", "paper_id"])
        for e in entries:
            pid = e.get("dc:identifier", "").replace("SCOPUS_ID:", "")
            authors = e.get("author", [])
            if not isinstance(authors, list):
                authors = [authors]
            for a in authors:
                if a.get("authid"):
                    writer.writerow([a.get("authid"), pid])

    conn.execute(f"COPY Author FROM '{authors_csv}' (HEADER=true);")
    conn.execute(f"COPY Paper FROM '{papers_csv}' (HEADER=true);")
    conn.execute(f"COPY WROTE FROM '{wrote_csv}' (HEADER=true);")

    print("Bulk load completed")


def insert_incremental(entries):
    """Insert new papers/authors with MERGE (no overwrite)."""
    for entry in entries:
        pid = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")
        title = entry.get("dc:title", "").replace('"', "'")
        doi = entry.get("prism:doi", "")
        pub = entry.get("prism:publicationName", "").replace('"', "'")
        year = entry.get("prism:coverDate", "").split("-")[0] if entry.get("prism:coverDate") else "NULL"
        keywords = entry.get("authkeywords", "")

        conn.execute("""
            MERGE (p:Paper {paper_id: $pid})
            SET p.title = $title, p.doi = $doi, p.publication_name = $pub,
                p.year = $year, p.keywords = $keywords
        """, {"pid": pid, "title": title, "doi": doi, "pub": pub,
              "year": int(year) if year != "NULL" else None, "keywords": keywords})

        authors = entry.get("author", [])
        if not isinstance(authors, list):
            authors = [authors]
        for a in authors:
            aid = a.get("authid", "")
            name = a.get("authname", "")
            if aid:
                conn.execute("""
                    MERGE (a:Author {author_id: $aid})
                    SET a.name = $name
                """, {"aid": aid, "name": name})
                conn.execute("""
                    MATCH (a:Author {author_id: $aid}), (p:Paper {paper_id: $pid})
                    MERGE (a)-[:WROTE]->(p)
                """, {"aid": aid, "pid": pid})
