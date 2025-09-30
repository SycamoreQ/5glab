
import pytest
import asyncio
from typing import List, Dict, Any
from graph.database.store import * 
from graph.database.schema import create_author_schema, create_paper_schema, wrote_relation
import kuzu
from kuzu import Database, Connection
import os
import tempfile
import shutil
import logging 


@pytest.fixture(scope="session")
def test_db_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    logging.info("db path tested")

    shutil.rmtree(temp_dir , ignore_errors= True)


@pytest.fixture(scope="session")
def setup_test_db(test_db_path):
    """Set up test database with schema and sample data."""
    db = Database(test_db_path)
    conn = Connection(db)
    
    # Create schema
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
        
        conn.execute("""
            CREATE REL TABLE CITES (FROM Paper TO Paper);
        """)
        
        # Insert test data
        test_authors = [
            ("auth1", "Alice Johnson"),
            ("auth2", "Bob Smith"),
            ("auth3", "Carol Williams")
        ]
        
        for auth_id, name in test_authors:
            conn.execute(
                "CREATE (a:Author {author_id: $1, name: $2})",
                [auth_id, name]
            )
        
        test_papers = [
            ("paper1", "Machine Learning Fundamentals", "10.1234/ml.2020", "AI Journal", 2020, ["machine learning", "AI"]),
            ("paper2", "Deep Learning Applications", "10.1234/dl.2021", "ML Conference", 2021, ["deep learning", "neural networks"]),
            ("paper3", "Natural Language Processing", "10.1234/nlp.2022", "NLP Journal", 2022, ["NLP", "transformers"]),
        ]
        
        for pid, title, doi, pub, year, keywords in test_papers:
            conn.execute(
                "CREATE (p:Paper {paper_id: $1, title: $2, doi: $3, publication_name: $4, year: $5, keywords: $6})",
                [pid, title, doi, pub, year, keywords]
            )
        
        relationships = [
            ("auth1", "paper1"),
            ("auth2", "paper1"),
            ("auth1", "paper2"),
            ("auth3", "paper3"),
        ]
        
        for auth_id, paper_id in relationships:
            conn.execute(
                "MATCH (a:Author {author_id: $1}), (p:Paper {paper_id: $2}) CREATE (a)-[:WROTE]->(p)",
                [auth_id, paper_id]
            )
        
        conn.execute(
            "MATCH (p1:Paper {paper_id: 'paper2'}), (p2:Paper {paper_id: 'paper1'}) CREATE (p1)-[:CITES]->(p2)"
        )
        
        conn.execute(
            "MATCH (p1:Paper {paper_id: 'paper3'}), (p2:Paper {paper_id: 'paper1'}) CREATE (p1)-[:CITES]->(p2)"
        )
        
    except Exception as e:
        print(f"Setup error: {e}")
        logging.error(e)
        raise
    
    yield conn
    
    conn.close()



class TestRetrieval: 

    @pytest.mark.asyncio
    async def test_get_papers_by_author(self, setup_test_db):
        """Test retrieving papers by author name."""
        papers = await get_papers_by_author("Alice Johnson")
        
        assert len(papers) == 2
        assert any(p["p.title"] == "Machine Learning Fundamentals" for p in papers)
        assert any(p["p.title"] == "Deep Learning Applications" for p in papers)
    
    @pytest.mark.asyncio
    async def test_get_authors_by_paper(self, setup_test_db):
        """Test retrieving authors by paper title."""
        authors = await get_authors_by_paper("Machine Learning Fundamentals")
        
        assert len(authors) == 2
        author_names = [a["a.name"] for a in authors]
        assert "Alice Johnson" in author_names
        assert "Bob Smith" in author_names
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id(self, setup_test_db):
        """Test retrieving a specific paper by ID."""
        paper = await get_paper_by_id("paper1")
        
        assert paper is not None
        assert paper["p.title"] == "Machine Learning Fundamentals"
        assert paper["p.year"] == 2020
        assert paper["p.doi"] == "10.1234/ml.2020"
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id_not_found(self, setup_test_db):
        """Test retrieving a non-existent paper."""
        paper = await get_paper_by_id("nonexistent")
        
        assert paper is None
    
    @pytest.mark.asyncio
    async def test_search_papers_by_title(self, setup_test_db):
        """Test searching papers by title substring."""
        papers = await search_papers_by_title("Learning")
        
        assert len(papers) >= 2
        titles = [p["p.title"] for p in papers]
        assert any("Machine Learning" in t for t in titles)
        assert any("Deep Learning" in t for t in titles)
    
    @pytest.mark.asyncio
    async def test_search_papers_case_insensitive(self, setup_test_db):
        """Test case-insensitive search."""
        papers = await search_papers_by_title("machine")
        
        assert len(papers) >= 1
        assert any("Machine Learning" in p["p.title"] for p in papers)
    
    @pytest.mark.asyncio
    async def test_search_authors_by_name(self, setup_test_db):
        """Test searching authors by name substring."""
        authors = await search_authors_by_name("Johnson")
        
        assert len(authors) >= 1
        assert any(a["a.name"] == "Alice Johnson" for a in authors)
    
    @pytest.mark.asyncio
    async def test_get_papers_by_year_range(self, setup_test_db):
        """Test retrieving papers within a year range."""
        papers = await get_papers_by_year_range(2020, 2021)
        
        assert len(papers) == 2
        years = [p["p.year"] for p in papers]
        assert all(2020 <= year <= 2021 for year in years)


class TestCitationRetrieval:
    """Test citation-related retrieval functions."""
    
    @pytest.mark.asyncio
    async def test_get_citation_count(self, setup_test_db):
        """Test getting citation count for a paper."""
        count = await get_citation_count("paper1")
        
        assert count == 2  # paper1 is cited by paper2 and paper3
    
    @pytest.mark.asyncio
    async def test_get_citation_count_zero(self, setup_test_db):
        """Test citation count for uncited paper."""
        count = await get_citation_count("paper3")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_most_cited_papers(self, setup_test_db):
        """Test retrieving most cited papers."""
        papers = await get_most_cited_papers(limit=5)
        
        assert len(papers) >= 1
        # paper1 should be the most cited
        assert papers[0]["cited.paper_id"] == "paper1"
        assert papers[0]["citation_count"] == 2


class TestEnhancedStore:
    """Test EnhancedStore async methods."""
    
    @pytest.fixture
    def store(self):
        """Create EnhancedStore instance."""
        return EnhancedStore(pool_size=5, enable_cache=True)
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_paper_info(self, setup_test_db, store):
        """Test comprehensive paper information retrieval."""
        info = await store.get_comprehensive_paper_info("paper1")
        
        assert info["paper_info"] is not None
        assert info["paper_info"]["p.title"] == "Machine Learning Fundamentals"
        assert len(info["authors"]) == 2
        assert info["citation_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_author_analytics(self, setup_test_db, store):
        """Test author analytics retrieval."""
        analytics = await store.get_author_analytics("auth1")
        
        assert analytics["author_info"]["a.name"] == "Alice Johnson"
        assert analytics["paper_count"] == 2
        assert len(analytics["papers"]) == 2
    
    @pytest.mark.asyncio
    async def test_advanced_search_by_title(self, setup_test_db, store):
        """Test advanced search with title keywords."""
        results = await store.advanced_search(
            title_keywords=["Machine", "Learning"],
            limit=10
        )
        
        assert results["total_found"] >= 1
        assert len(results["papers"]) >= 1
    
    @pytest.mark.asyncio
    async def test_advanced_search_by_year_range(self, setup_test_db, store):
        """Test advanced search with year range."""
        results = await store.advanced_search(
            year_range=(2020, 2021),
            limit=10
        )
        
        assert results["total_found"] == 2
        years = [p["p.year"] for p in results["papers"]]
        assert all(2020 <= year <= 2021 for year in years)
    
    @pytest.mark.asyncio
    async def test_track_keyword_temporal_trend(self, setup_test_db, store):
        """Test keyword temporal trend tracking."""
        trends = await store.track_keyword_temporal_trend("learning", 2020, 2022)
        
        assert len(trends) >= 1
        assert all("p.year" in t for t in trends)
        assert all("keyword_count" in t for t in trends)


class TestErrorHandling:
    """Test error handling in retrieval functions."""
    
    @pytest.mark.asyncio
    async def test_empty_author_name(self, setup_test_db):
        """Test handling of empty author name."""
        papers = await get_papers_by_author("")
        
        assert papers == []
    
    @pytest.mark.asyncio
    async def test_nonexistent_author(self, setup_test_db):
        """Test querying non-existent author."""
        papers = await get_papers_by_author("Nonexistent Author")
        
        assert papers == []
    
    @pytest.mark.asyncio
    async def test_invalid_year_range(self, setup_test_db):
        """Test invalid year range (start > end)."""
        papers = await get_papers_by_year_range(2022, 2020)
        
        assert papers == []
