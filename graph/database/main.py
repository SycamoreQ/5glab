import asyncio
from .crawler import setup_schema
from .crawler_2 import load_kaggle_json, load_cites_edges_only 
from .store import EnhancedStore
from .vector.inject import fetch_and_vectorize_pdf

KAGGLE_JSON = "/Users/kaushikmuthukumar/Downloads/dblp.v12.json"

async def main():
    print("Step 1: Setting up Neo4j schema constraints and indexes...")
    setup_schema()

    print("\nStep 2: Loading Kaggle citation network nodes/authors into Neo4j (no CITES yet)...")
    #load_kaggle_json(KAGGLE_JSON)

    print("\nStep 3: Loading CITES edges (references) into Neo4j...")
    load_cites_edges_only(KAGGLE_JSON)  # This adds CITES edges in a separate pass

    store = EnhancedStore()

    print("\nStep 4: Querying for sample author papers in DB...")
    authors = await store.get_all_authors(limit=1)
    if authors:
        sample_author_name = authors[0].get("name", "")
        print(f"Sample author name: {sample_author_name}")
        papers = await store.get_papers_by_author(sample_author_name)
        print(f"Papers by '{sample_author_name}': {papers}")
    else:
        print("No authors found in DB; skipping author query test.")

    print("\nStep 5: Injecting sample papers into ChromaDB for embedding...")
    papers_to_inject = await store.get_all_papers(limit=3)
    for paper in papers_to_inject:
        fetch_and_vectorize_pdf(paper)

    print("\nAll steps completed.")

if __name__ == "__main__":
    asyncio.run(main())
