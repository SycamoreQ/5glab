import pytest
from testing.db_setup import setup_test_db


class TestRetrieval:

    def test_get_papers_by_author(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (a:Author {name: 'Alice'})-[:WROTE]->(p:Paper)
            RETURN p.title
        """)
        titles = [row[0] for row in result]
        assert "Graph Databases 101" in titles
        assert len(titles) == 1

    def test_get_authors_by_paper(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper {title: 'Advanced RAG'})<-[:WROTE]-(a:Author)
            RETURN a.name
        """)
        authors = [row[0] for row in result]
        assert "Bob" in authors
        assert len(authors) == 1

    def test_get_paper_by_id(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper {paper_id: 'P1'})
            RETURN p.title, p.year
        """)
        row = result[0]
        assert row[0] == "Graph Databases 101"
        assert row[1] == 2020

    def test_get_paper_by_id_not_found(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper {paper_id: 'P999'})
            RETURN p
        """)
        assert len(result) == 0

    def test_search_papers_by_title(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE p.title CONTAINS 'Graph'
            RETURN p.title
        """)
        titles = [row[0] for row in result]
        assert "Graph Databases 101" in titles

    def test_search_papers_case_insensitive(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE LOWER(p.title) CONTAINS 'rag'
            RETURN p.title
        """)
        titles = [row[0] for row in result]
        assert "Advanced RAG" in titles

    def test_search_authors_by_name(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (a:Author)
            WHERE a.name = 'Alice'
            RETURN a.name
        """)
        names = [row[0] for row in result]
        assert names == ["Alice"]

    def test_get_papers_by_year_range(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE p.year >= 2020 AND p.year <= 2021
            RETURN p.title
        """)
        titles = {row[0] for row in result}
        assert "Graph Databases 101" in titles
        assert "Advanced RAG" in titles


class TestCitationRetrieval:

    def test_get_citation_count(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p1:Paper {paper_id: 'P1'})<-[:CITES]-(p2:Paper)
            RETURN COUNT(p2)
        """)
        count = result[0][0]
        assert count == 1

    def test_get_citation_count_zero(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p1:Paper {paper_id: 'P2'})<-[:CITES]-(p2:Paper)
            RETURN COUNT(p2)
        """)
        count = result[0][0]
        assert count == 0

    def test_get_most_cited_papers(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p1:Paper)<-[:CITES]-(p2:Paper)
            RETURN p1.title, COUNT(p2) AS citations
            ORDER BY citations DESC
            LIMIT 1
        """)
        row = result[0]
        assert row[0] == "Graph Databases 101"
        assert row[1] == 1


class TestErrorHandling:

    def test_empty_author_name(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (a:Author {name: ''}) RETURN a
        """)
        assert len(result) == 0

    def test_nonexistent_author(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (a:Author {name: 'Charlie'}) RETURN a
        """)
        assert len(result) == 0

    def test_invalid_year_range(self, setup_test_db):
        conn = setup_test_db
        result = conn.execute("""
            MATCH (p:Paper)
            WHERE p.year >= 2025 AND p.year <= 2030
            RETURN p
        """)
        assert len(result) == 0
