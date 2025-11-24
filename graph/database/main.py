import os
import asyncio
from crawler import fetch_papers, bulk_load, setup_schema, db_is_empty
from store import EnhancedStore
from vector.inject import fetch_and_vectorize_pdf
from utils.config import BaseConfig

API_KEY = BaseConfig.elsevier_api_key
SUBJECT = "machine learning"
COUNT = 5

async def main():
    print("Step 1: Setting up Neo4j schema constraints and indexes...")
    setup_schema()

    print(f"\nStep 2: Fetching {COUNT} papers on subject '{SUBJECT}' from Elsevier API...")
    entries = fetch_papers(SUBJECT, count=COUNT)
    if not entries:
        print("Fetch returned no entries - check your API key and internet connection.")
        return

    print("\nStep 3: Bulk loading papers and authors into Neo4j...")
    bulk_load(entries)

    print("\nStep 4: Checking if database contains data after loading...")
    if db_is_empty():
        print("DB appears empty after bulk load. Aborting.")
        return

    store = EnhancedStore(poolsize=3)

    # Test querying papers by author for validation
    sample_author = entries[0].get("author", [])
    if sample_author:
        sample_author_name = sample_author[0].get("authname", "")
        print(f"\nStep 5: Querying papers written by author '{sample_author_name}'")
        papers = await store.get_papers_by_author(sample_author_name)
        print(f"Papers by '{sample_author_name}': {papers}")
    else:
        print("Entry had no authors; skipping author query test.")

    # Inject sample paper nodes into ChromaDB using fetch and vectorize
    print("\nStep 6: Injecting sample papers into ChromaDB for embedding...")
    papers_to_inject = await store.getallpapers(limit=3)
    for paper in papers_to_inject:
        fetch_and_vectorize_pdf(paper)

    print("\nAll steps completed.")


if __name__ == "__main__":
    asyncio.run(main())
