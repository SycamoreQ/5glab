import os
import asyncio
from .crawler import fetch_papers, bulk_load_with_details, setup_schema, db_is_empty , fetch_papers_paginated
from .store import EnhancedStore
from .vector.inject import fetch_and_vectorize_pdf
from utils.config import BaseConfig

API_KEY = BaseConfig.elsevier_api_key
SUBJECT = "COMP"
COUNT = 200

async def main():
    print("Step 1: Setting up Neo4j schema constraints and indexes...")
    setup_schema()

    print(f"\nStep 2: Fetching {COUNT} papers on subject '{SUBJECT}' from Elsevier API...")
    entries = fetch_papers_paginated(subject=SUBJECT)
    print(entries)
    if not entries:
        print("Fetch returned no entries - check your API key and internet connection.")
        return

    print("\nStep 3: Bulk loading papers and authors into Neo4j (with detailed author fetch)") 
    bulk_load_with_details(entries) 

    print("\nStep 4: Checking if database contains data after loading...")
    if db_is_empty():
        print("DB appears empty after bulk load. Aborting.")
        return

    store = EnhancedStore()

    print("\nStep 5: Querying for sample author papers in DB...")
    authors = await store.get_all_authors(limit=1)
    if authors:
        sample_author_name = authors[0].get("name", "")
        if sample_author_name:
            print(f"Sample author name: {sample_author_name}")
            papers = await store.get_papers_by_author(sample_author_name)
            print(f"Papers by '{sample_author_name}': {papers}")
        else:
            print("Author in DB has no name. Skipping author-paper query test.")
    else:
        print("No authors found in DB; skipping author query test.")

    print("\nStep 6: Injecting sample papers into ChromaDB for embedding...")
    papers_to_inject = await store.get_all_papers(limit=3)
    for paper in papers_to_inject:
        fetch_and_vectorize_pdf(paper)

    print("\nAll steps completed.")

if __name__ == "__main__":
    asyncio.run(main())
