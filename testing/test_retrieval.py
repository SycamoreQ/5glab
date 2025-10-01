import pytest
import asyncio
import os
import tempfile
import shutil
from kuzu import Database, Connection


# ============================================
# CRITICAL: Test Database Setup
# ============================================

@pytest.fixture(scope="session")
def test_db_path():
    """Create isolated test database directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_kuzu_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_database(test_db_path):
    """Create and populate test database with schema and data."""
    db = Database(test_db_path)
    conn = Connection(db)
    
    try:
        # 1. CREATE SCHEMA
        print("Creating schema...")
        
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Author (
                author_id STRING PRIMARY KEY,
                name STRING
            );
        """)
        
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Paper (
                paper_id STRING PRIMARY KEY,
                title STRING,
                doi STRING,
                publication_name STRING,
                year INT,
                keywords STRING[]
            );
        """)
        
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS WROTE (FROM Author TO Paper);
        """)
        
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CITES (FROM Paper TO Paper);
        """)
        
        # 2. INSERT TEST DATA
        print("Inserting test data...")
        
        # Insert Authors
        test_authors = [
            ("auth1", "Alice Johnson"),
            ("auth2", "Bob Smith"),
            ("auth3", "Carol Williams"),
            ("auth4", "David Brown")
        ]
        
        for auth_id, name in test_authors:
            conn.execute(
                "CREATE (a:Author {author_id: $auth_id, name: $name})",
                {"auth_id": auth_id, "name": name}
            )
        
        # Insert Papers
        test_papers = [
            {
                "paper_id": "paper1",
                "title": "Machine Learning Fundamentals",
                "doi": "10.1234/ml.2020",
                "publication_name": "AI Journal",
                "year": 2020,
                "keywords": ["machine learning", "AI", "fundamentals"]
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
                "title": "Natural Language Processing Advances",
                "doi": "10.1234/nlp.2022",
                "publication_name": "NLP Journal",
                "year": 2022,
                "keywords": ["NLP", "transformers", "BERT"]
            },
            {
                "paper_id": "paper4",
                "title": "Graph Neural Networks",
                "doi": "10.1234/gnn.2023",
                "publication_name": "Graph Conference",
                "year": 2023,
                "keywords": ["graphs", "neural networks", "GNN"]
            }
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
        
        # 3. CREATE AUTHORSHIP RELATIONSHIPS
        authorships = [
            ("auth1", "paper1"),  # Alice -> ML Fundamentals
            ("auth2", "paper1"),  # Bob -> ML Fundamentals (co-author)
            ("auth1", "paper2"),  # Alice -> Deep Learning
            ("auth3", "paper3"),  # Carol -> NLP
            ("auth4", "paper4"),  # David -> GNN
            ("auth1", "paper4"),  # Alice -> GNN (co-author)
        ]
        
        for auth_id, paper_id in authorships:
            conn.execute("""
                MATCH (a:Author {author_id: $auth_id}), 
                      (p:Paper {paper_id: $paper_id})
                CREATE (a)-[:WROTE]->(p)
            """, {"auth_id": auth_id, "paper_id": paper_id})
        
        # 4. CREATE CITATION RELATIONSHIPS
        citations = [
            ("paper2", "paper1"),  # Deep Learning cites ML Fundamentals
            ("paper3", "paper1"),  # NLP cites ML Fundamentals
            ("paper4", "paper2"),  # GNN cites Deep Learning
            ("paper4", "paper3"),  # GNN cites NLP
        ]
        
        for citing, cited in citations:
            conn.execute("""
                MATCH (p1:Paper {paper_id: $citing}),
                      (p2:Paper {paper_id: $cited})
                CREATE (p1)-[:CITES]->(p2)
            """, {"citing": citing, "cited": cited})
        
        print("Test database setup complete!")
        
    except Exception as e:
        print(f"Error setting up test database: {e}")
        raise
    finally:
        conn.close()
    
    return test_db_path


@pytest.fixture(autouse=True)
def mock_db_path(monkeypatch, test_db_path):
    """Override DB_PATH in all modules to use test database."""
    # Patch the DB_PATH constant in your modules
    monkeypatch.setattr("graph.database.store.DB_PATH", test_db_path)
    monkeypatch.setattr("graph.database.schema.DB_PATH", test_db_path)
    
    # Also patch the global connection pool if it exists
    import graph.database.store as store_module
    store_module._connection_pool = store_module.ConnectionPool(test_db_path)


# ============================================
# ALTERNATIVE: Manual Setup/Teardown
# ============================================

@pytest.fixture
async def db_with_data(test_db_path):
    """
    Alternative: Explicitly setup and teardown for each test.
    Use this if you want fresh data for each test.
    """
    db = Database(test_db_path)
    conn = Connection(db)
    
    # Setup code here (same as above)
    
    yield conn
    
    # Teardown: optionally clear data
    try:
        conn.execute("MATCH (a:Author) DETACH DELETE a")
        conn.execute("MATCH (p:Paper) DETACH DELETE p")
    except:
        pass
    finally:
        conn.close()


# ============================================
# VERIFICATION TESTS
# ============================================

class TestDatabaseSetup:
    """Verify test database is properly set up."""
    
    @pytest.mark.asyncio
    async def test_database_has_authors(self, test_database):
        """Verify authors were loaded."""
        from graph.database.store import get_all_authors
        
        authors = await get_all_authors()
        assert len(authors) >= 4
        author_names = [a.get("a.name") for a in authors]
        assert "Alice Johnson" in author_names
    
    @pytest.mark.asyncio
    async def test_database_has_papers(self, test_database):
        """Verify papers were loaded."""
        from graph.database.store import get_all_papers
        
        papers = await get_all_papers()
        assert len(papers) >= 4
        titles = [p.get("p.title") for p in papers]
        assert "Machine Learning Fundamentals" in titles
    
    @pytest.mark.asyncio
    async def test_database_has_relationships(self, test_database):
        """Verify relationships were created."""
        from graph.database.store import get_papers_by_author
        
        papers = await get_papers_by_author("Alice Johnson")
        assert len(papers) >= 2  # Alice wrote at least 2 papers
    
    @pytest.mark.asyncio
    async def test_database_has_citations(self, test_database):
        """Verify citation relationships exist."""
        from graph.database.store import get_citation_count
        
        count = await get_citation_count("paper1")
        assert count >= 2  # paper1 is cited by paper2 and paper3


# ============================================
# ACTUAL RETRIEVAL TESTS
# ============================================

class TestRetrieval:
    """Test retrieval functions with loaded database."""
    
    @pytest.mark.asyncio
    async def test_get_papers_by_author(self, test_database):
        """Test retrieving papers by author."""
        from graph.database.store import get_papers_by_author
        
        papers = await get_papers_by_author("Alice Johnson")
        
        assert len(papers) >= 2
        titles = [p.get("p.title") for p in papers]
        assert "Machine Learning Fundamentals" in titles
        assert "Deep Learning Applications" in titles
    
    @pytest.mark.asyncio
    async def test_get_authors_by_paper(self, test_database):
        """Test retrieving authors of a paper."""
        from graph.database.store import get_authors_by_paper
        
        authors = await get_authors_by_paper("Machine Learning Fundamentals")
        
        assert len(authors) == 2
        names = [a.get("a.name") for a in authors]
        assert "Alice Johnson" in names
        assert "Bob Smith" in names
    
    @pytest.mark.asyncio
    async def test_search_papers(self, test_database):
        """Test paper search functionality."""
        from graph.database.store import search_papers_by_title
        
        papers = await search_papers_by_title("Learning")
        
        assert len(papers) >= 2
        # Should find both "Machine Learning" and "Deep Learning"
    
    @pytest.mark.asyncio
    async def test_citation_count(self, test_database):
        """Test getting citation count."""
        from graph.database.store import get_citation_count
        
        count = await get_citation_count("paper1")
        
        assert count == 2  # Cited by paper2 and paper3
    
    @pytest.mark.asyncio
    async def test_year_range_query(self, test_database):
        """Test querying papers by year range."""
        from graph.database.store import get_papers_by_year_range
        
        papers = await get_papers_by_year_range(2020, 2021)
        
        assert len(papers) == 2
        years = [p.get("p.year") for p in papers]
        assert all(2020 <= y <= 2021 for y in years)


# ============================================
# MANUAL SETUP EXAMPLE
# ============================================

def manual_setup_example():
    """
    If you want to manually set up the database before running tests,
    you can use this approach instead.
    """
    db_path = "test_research_db"
    db = Database(db_path)
    conn = Connection(db)
    
    # Your setup code...
    
    conn.close()
    
    # Then run: pytest with DB_PATH environment variable
    # DB_PATH=test_research_db pytest tests/


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])