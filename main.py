from graph.database.crawler import fetch_papers , save_to_csv
from graph.database.schema import create_schema, load_csv_into_kuzu
from kuzu import Database, Connection

DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)

def main():
    subjects = ["COMP", "MATH", "PHYS"] 
    all_entries = []
    for subj in subjects:
        entries = fetch_papers(subj, count=10)
        all_entries.extend(entries)

    save_to_csv(all_entries, "papers.csv", "authors.csv", "wrote.csv")
    print("✅ CSVs created")

    create_schema()
    load_csv_into_kuzu()

    result = conn.execute("MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name, p.title LIMIT 5;")
    for row in result:
        print(row)


if __name__ == "__main__":
    main()
