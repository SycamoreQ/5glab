import kuzu 
from kuzu import Database, Connection

DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)


def create_schema():
    try:
        conn.execute("""
            CREATE NODE TABLE Author (
                author_id STRING PRIMARY KEY,
                name STRING
            );
        """)
        conn.execute("""
            CREATE NODE TABLE Paper (
                paper_id STRING PRIMARY KEY,
                title STRING,
                doi STRING,
                publication_name STRING,
                year INT,
                keywords STRING[]
            );
        """)
        conn.execute("""
            CREATE REL TABLE WROTE (FROM Author TO Paper);
        """)
        print("Schema created successfully")
    except Exception as e:
        print(f"Schema creation failed: {e}")



def load_csv_into_kuzu():
    conn.execute("COPY Author FROM 'authors.csv';")
    conn.execute("COPY Paper FROM 'papers.csv';")
    conn.execute("COPY WROTE FROM 'wrote.csv';")