"""
Pytest configuration and fixtures for testing.
Place this file in your tests/ directory.
"""

import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path
from kuzu import Database, Connection

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_path():
    """
    Create isolated test database path.
    
    IMPORTANT: Returns a path that doesn't exist yet.
    KuzuDB will create the directory when Database() is initialized.
    """
    # Create parent directory
    temp_parent = tempfile.mkdtemp(prefix="pytest_kuzu_")
    
    # DB path is a subdirectory (doesn't exist yet)
    db_path = os.path.join(temp_parent, "test_research_db")
    
    print(f"\nTest DB path: {db_path}")
    
    yield db_path
    
    # Cleanup after all tests
    print(f"\nCleaning up test database at {temp_parent}")
    shutil.rmtree(temp_parent, ignore_errors=True)


@pytest.fixture(scope="session")
def test_database(test_db_path):
    """
    Initialize test database with schema and sample data.
    This runs once for all tests.
    """
    print(f"\n{'='*60}")
    print("Setting up test database...")
    print(f"{'='*60}")
    
    # Initialize database (KuzuDB creates the directory)
    db = Database(test_db_path)
    conn = Connection(db)
    
    try:
        # CREATE SCHEMA
        print("Creating schema...")
        
        conn.execute("""
            CREATE NODE TABLE Author (
                author_id STRING PRIMARY KEY,
                name STRING
            )
        """)
        
        conn.execute("""
            CREATE NODE TABLE Paper (
                paper_id STRING PRIMARY KEY,
                title STRING,
                doi STRING,
                publication_name STRING,
                year INT,
                keywords STRING[]
            )
        """)
        
        conn.execute("CREATE REL TABLE WROTE (FROM Author TO Paper)")
        conn.execute("CREATE REL TABLE CITES (FROM Paper TO Paper)")
        
        print("✓ Schema created")
        
        # INSERT TEST DATA
        print("Inserting test data...")
        
        # Authors
        test_authors = [
            ("auth1", "Alice Johnson"),
            ("auth2", "Bob Smith"),
            ("auth3", "Carol Williams"),
            ("auth4", "David Brown"),
        ]
        
        for auth_id, name in test_authors:
            conn.execute(
                "CREATE (a:Author {author_id: $auth_id, name: $name})",
                {"auth_id": auth_id, "name": name}
            )
        
        # Papers
        test_papers = [
            {
                "paper_id": "paper1",
                "title": "Machine Learning Fundamentals",
                "doi": "10.1234/ml.2020",
                "publication_name": "AI Journal",
                "year": 2020,
                "keywords": ["machine learning", "AI"]
            },
            {
                "paper_id": "paper2",
                "title": "Deep Learning Applications",
                "doi": "10.1234/dl.2021",
                "publication_name": "ML Conference",
                "year": 2021,
                "keywords": ["deep learning", "neural networks"]
            },
            {
                "paper_id": "paper3",
                "title": "Natural Language Processing",
                "doi": "10.1234/nlp.2022",
                "publication_name": "NLP Journal",
                "year": 2022,
                "keywords": ["NLP", "transformers"]
            },
            {
                "paper_id": "paper4",
                "title": "Graph Neural Networks",
                "doi": "10.1234/gnn.2023",
                "publication_name": "Graph Conference",
                "year": 2023,
                "keywords": ["GNN", "graphs"]
            },
        ]
        
        for paper in test_papers:
            conn.execute("""
                CREATE (p:Paper {
                    paper_id: $paper_id,
                    title: $title,
                    doi: $doi,
                    publication_name: $publication_name,
                    year: $year,
                    keywords: $keywords
                })
            """, paper)
        
        # Relationships
        authorships = [
            ("auth1", "paper1"),
            ("auth2", "paper1"),
            ("auth1", "paper2"),
            ("auth3", "paper3"),
            ("auth4", "paper4"),
            ("auth1", "paper4"),
        ]
        
        for auth_id, paper_id in authorships:
            conn.execute("""
                MATCH (a:Author {author_id: $auth_id}), 
                      (p:Paper {paper_id: $paper_id})
                CREATE (a)-[:WROTE]->(p)
            """, {"auth_id": auth_id, "paper_id": paper_id})
        
        citations = [
            ("paper2", "paper1"),
            ("paper3", "paper1"),
            ("paper4", "paper2"),
        ]
        
        for citing, cited in citations:
            conn.execute("""
                MATCH (p1:Paper {paper_id: $citing}),
                      (p2:Paper {paper_id: $cited})
                CREATE (p1)-[:CITES]->(p2)
            """, {"citing": citing, "cited": cited})
        
        print("✓ Test data loaded")
        
        # Verify
        result = conn.execute("MATCH (a:Author) RETURN COUNT(a) AS count")
        count = result.get_next()[0]
        print(f"✓ Loaded {count} authors")
        
        result = conn.execute("MATCH (p:Paper) RETURN COUNT(p) AS count")
        count = result.get_next()[0]
        print(f"✓ Loaded {count} papers")
        
        print(f"{'='*60}")
        print("Test database ready!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise
    finally:
        conn.close()
    
    # Return the path so tests can use it
    return test_db_path


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, test_db_path):
    """
    Automatically patch DB_PATH for all tests.
    This ensures tests use the test database, not production.
    """
    # Patch the DB_PATH constant in your modules
    import graph.database.store as store_module
    import graph.database.schema as schema_module
    
    monkeypatch.setattr(store_module, "DB_PATH", test_db_path)
    monkeypatch.setattr(schema_module, "DB_PATH", test_db_path)
    
    # Recreate connection pool with test path
    store_module._connection_pool = store_module.ConnectionPool(test_db_path)
    
    # Set environment variable as well
    monkeypatch.setenv("DB_PATH", test_db_path)


# Optional: Fixtures for common test data
@pytest.fixture
def sample_author_id():
    """Common author ID for tests."""
    return "auth1"


@pytest.fixture
def sample_paper_id():
    """Common paper ID for tests."""
    return "paper1"


@pytest.fixture
def sample_author_name():
    """Common author name for tests."""
    return "Alice Johnson"


@pytest.fixture
def sample_paper_title():
    """Common paper title for tests."""
    return "Machine Learning Fundamentals"