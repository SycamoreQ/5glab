# db_setup.py
import pytest
import kuzu

@pytest.fixture(scope="function")
def test_db_path(tmp_path):
    """
    Creates a temporary database file path inside pytest's tmp_path directory.
    """
    return str(tmp_path / "test_db.kuzu")


@pytest.fixture(scope="function")
def setup_test_db(test_db_path):
    """
    Initializes a temporary Kùzu database, sets up schema, inserts sample data,
    and yields a connection object for tests.
    """
    db = kuzu.Database(test_db_path)
    conn = kuzu.Connection(db)

    # --- Schema ---
    conn.execute("""
        CREATE NODE TABLE Author(author_id STRING, name STRING, PRIMARY KEY(author_id));
    """)
    conn.execute("""
        CREATE NODE TABLE Paper(paper_id STRING, title STRING, year INT64, PRIMARY KEY(paper_id));
    """)
    conn.execute("""
        CREATE REL TABLE WROTE(FROM Author TO Paper);
    """)
    conn.execute("""
        CREATE REL TABLE CITES(FROM Paper TO Paper);
    """)

    # --- Sample Data ---
    conn.execute("INSERT INTO Author(author_id, name) VALUES ('A1', 'Alice');")
    conn.execute("INSERT INTO Author(author_id, name) VALUES ('A2', 'Bob');")

    conn.execute("INSERT INTO Paper(paper_id, title, year) VALUES ('P1', 'Graph Databases 101', 2020);")
    conn.execute("INSERT INTO Paper(paper_id, title, year) VALUES ('P2', 'Advanced RAG', 2021);")

    conn.execute("""
        INSERT INTO WROTE VALUES (
            (SELECT a FROM Author a WHERE a.author_id = 'A1'),
            (SELECT p FROM Paper p WHERE p.paper_id = 'P1')
        );
    """)
    conn.execute("""
        INSERT INTO WROTE VALUES (
            (SELECT a FROM Author a WHERE a.author_id = 'A2'),
            (SELECT p FROM Paper p WHERE p.paper_id = 'P2')
        );
    """)
    conn.execute("""
        INSERT INTO CITES VALUES (
            (SELECT p1 FROM Paper p1 WHERE p1.paper_id = 'P2'),
            (SELECT p2 FROM Paper p2 WHERE p2.paper_id = 'P1')
        );
    """)

    yield conn

    # --- Teardown ---
    conn.close()
