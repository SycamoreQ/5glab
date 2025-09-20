from graph.database.crawler import fetch_papers , bulk_load , insert_incremental , db_is_empty
from graph.database.schema import create_schema, load_csv_into_kuzu
from kuzu import Database, Connection

DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)

def main():
    create_schema()

    subjects = ["COMP", "MATH", "PHYS"]
    all_entries = []
    for subj in subjects:
        all_entries.extend(fetch_papers(subj, count=5))

    if db_is_empty():
        print("⚡ First run → using bulk load")
        bulk_load(all_entries)
    else:
        print("🔄 Incremental mode → inserting new data")
        insert_incremental(all_entries)

    result = conn.execute("MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name, p.title LIMIT 5;")
    for row in result:
        print(row)


if __name__ == "__main__":
    main()
