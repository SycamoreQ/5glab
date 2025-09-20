import csv
import requests
from kuzu import Database, Connection


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


def save_to_csv(entries, paper_file, author_file, wrote_file):
    with open(paper_file, "w", newline="", encoding="utf-8") as pf, \
         open(author_file, "w", newline="", encoding="utf-8") as af, \
         open(wrote_file, "w", newline="", encoding="utf-8") as wf:

        paper_writer = csv.writer(pf)
        author_writer = csv.writer(af)
        wrote_writer = csv.writer(wf)

        # Headers for COPY
        paper_writer.writerow(["paper_id", "title", "doi", "publication_name", "year", "keywords"])
        author_writer.writerow(["author_id", "name"])
        wrote_writer.writerow(["author_id", "paper_id"])

        seen_authors = set()

        for entry in entries:
            pid = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")
            title = entry.get("dc:title", "").replace('"', "'")
            doi = entry.get("prism:doi", "")
            pub = entry.get("prism:publicationName", "").replace('"', "'")
            year = entry.get("prism:coverDate", "").split("-")[0] if entry.get("prism:coverDate") else ""
            keywords = entry.get("authkeywords", "")

            paper_writer.writerow([pid, title, doi, pub, year, keywords])

            authors = entry.get("author", [])
            if not isinstance(authors, list):
                authors = [authors]
            for a in authors:
                aid = a.get("authid", "")
                name = a.get("authname", "")
                if aid and aid not in seen_authors:
                    author_writer.writerow([aid, name])
                    seen_authors.add(aid)
                if aid:
                    wrote_writer.writerow([aid, pid])