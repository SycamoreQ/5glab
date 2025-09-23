import kuzu 
from kuzu import Database, Connection
import logging
import os   

DB_PATH = "research_db"  
db = Database(DB_PATH)
conn = Connection(db)


def create_author_schema():
    """ Create author schema"""

    try:
        conn.execute("""
            CREATE NODE TABLE Author (
                author_id STRING PRIMARY KEY,
                name STRING
            );
        """)
        print("author schema created successfully")

    except Exception as e:
        logging.error(f"Schema creation failed: {e}")
        print(f"Author schema creation failed: {e}")


def create_paper_schema():
    """Create paper schema """

    try:
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

        print("paper schema created successfully")

    except Exception as e: 
        logging.error(f"Schema creation failed: {e}")
        print(f"Paper creation schema failed: {e}")


def wrote_relation():
    """Create WROTE relation"""

    try:
        conn.execute("""
            CREATE REL TABLE WROTE (FROM Author TO Paper);
        """)

        print("relation WROTE created")

    except Exception as e: 
        logging.error(f"relation wrote creation failed: {e}")
        print(f"relation wrote creation failed: {e}") 


def cites_relation():
    """"""
    try:         
        conn.execute("""
                CREATE REL TABLE CITES (FROM Paper TO Paper);

            """
        )

        print("relation CITED created successfully")

    except Exception as e: 
        logging.error(f"relation wrote creation failed: {e}")
        print(f"relation wrote creation failed: {e}") 



def similar_to_relation():
    try:
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(
                    FROM PAPER TO PAPER)
                    similarity_score FLOAT
                     """)
        print("relation SIMILAR_TO created successfully")

    except Exception as e:
        logging.error(f"relation similar_to failed to be created: {e}")
        print(f"Schema creation failed: {e}")



def drop_schema():
    """Drop all tables - useful for testing/resetting"""
    try:
        # Drop relationship tables first (due to dependencies)
        tables_to_drop = [
            "CITES", "WROTE",
            "Paper", "Author"
        ]
        
        for table in tables_to_drop:
            try:
                conn.execute(f"DROP TABLE {table};")
            except Exception as e:
                # Table might not exist, continue
                logging.warning(f"Could not drop table {table}: {e}")
        
        conn.commit()
        logging.info("Schema dropped successfully")
        print("Schema dropped successfully")
        return True
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Schema drop failed: {e}")
        print(f"Schema drop failed: {e}")
        return False




def load_csv_into_kuzu():
    """Load CSV files with proper transaction control and error handling"""
    try:
        # Check if CSV files exist
        csv_files = {
            'authors.csv': 'Author',
            'papers.csv': 'Paper', 
            'wrote.csv': 'WROTE',
            #'cited.csv': 'CITES'  Uncomment if using citations
        }
        
        for csv_file, table_name in csv_files.items():
            if not os.path.exists(csv_file):
                logging.warning(f"CSV file {csv_file} not found, skipping...")
                continue
            
            logging.info(f"Loading {csv_file} into {table_name}")
            conn.execute(f"COPY {table_name} FROM '{csv_file}';")

        conn.commit()
        logging.info("CSV data loaded successfully")
        print("CSV data loaded successfully")
        return True
    
    except Exception as e:
        conn.rollback()
        logging.error(f"CSV loading failed: {e}")
        print(f"CSV loading failed: {e}")
        return False
    



#placeholder for future use
def index_tables():
    """Create indexes on frequently queried fields"""
    try:
        conn.execute("CREATE INDEX ON :Author(name);")
        conn.execute("CREATE INDEX ON :Paper(title);")
        conn.execute("CREATE INDEX ON :Paper(doi);")
        conn.execute("CREATE INDEX ON :Paper(year);")
        conn.execute("CREATE INDEX ON :Paper(keywords);")
        conn.commit()
        logging.info("Indexes created successfully")
        print("Indexes created successfully")
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Index creation failed: {e}")
        print(f"Index creation failed: {e}")
        return False


    