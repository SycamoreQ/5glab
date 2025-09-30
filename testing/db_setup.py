TEST_DB_PATH = "test_research_db"

import sys
import os
from kuzu import Database, Connection
import shutil

def setup_test_database():
    """Create and populate test database."""
    
    # Remove existing test database if it exists
    if os.path.exists(TEST_DB_PATH):
        print(f"Removing existing test database at {TEST_DB_PATH}...")
        shutil.rmtree(TEST_DB_PATH)
    
    print(f"Creating new test database at {TEST_DB_PATH}...")
    db = Database(TEST_DB_PATH)
    conn = Connection(db)
    
    try:
        # ========================================
        # STEP 1: CREATE SCHEMA
        # ========================================
        print("\n[1/4] Creating schema...")
        
        conn.execute("""
            CREATE NODE TABLE Author (
                author_id STRING PRIMARY KEY,
                name STRING
            );
        """)
        print("  ✓ Created Author table")
        
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
        print("  ✓ Created Paper table")
        
        conn.execute("""
            CREATE REL TABLE WROTE (FROM Author TO Paper);
        """)
        print("  ✓ Created WROTE relationship")
        
        conn.execute("""
            CREATE REL TABLE CITES (FROM Paper TO Paper);
        """)
        print("  ✓ Created CITES relationship")
        
        # ========================================
        # STEP 2: INSERT AUTHORS
        # ========================================
        print("\n[2/4] Inserting test authors...")
        
        test_authors = [
            ("auth1", "Alice Johnson"),
            ("auth2", "Bob Smith"),
            ("auth3", "Carol Williams"),
            ("auth4", "David Brown"),
            ("auth5", "Eve Davis")
        ]
        
        for auth_id, name in test_authors:
            conn.execute(
                "CREATE (a:Author {author_id: $auth_id, name: $name})",
                {"auth_id": auth_id, "name": name}
            )
            print(f"  ✓ Added {name}")
        
        # ========================================
        # STEP 3: INSERT PAPERS
        # ========================================
        print("\n[3/4] Inserting test papers...")
        
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
                "title": "Graph Neural Networks Survey",
                "doi": "10.1234/gnn.2023",
                "publication_name": "Graph Conference",
                "year": 2023,
                "keywords": ["graphs", "neural networks", "GNN"]
            },
            {
                "paper_id": "paper5",
                "title": "Reinforcement Learning in Practice",
                "doi": "10.1234/rl.2023",
                "publication_name": "RL Workshop",
                "year": 2023,
                "keywords": ["reinforcement learning", "RL", "agents"]
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
            print(f"  ✓ Added {paper['title']}")
        
        # ========================================
        # STEP 4: CREATE RELATIONSHIPS
        # ========================================
        print("\n[4/4] Creating relationships...")
        
        # Authorship relationships
        print("  Creating WROTE relationships...")
        authorships = [
            ("auth1", "paper1"),  # Alice -> ML Fundamentals
            ("auth2", "paper1"),  # Bob -> ML Fundamentals (co-author)
            ("auth1", "paper2"),  # Alice -> Deep Learning
            ("auth3", "paper3"),  # Carol -> NLP
            ("auth4", "paper4"),  # David -> GNN
            ("auth1", "paper4"),  # Alice -> GNN (co-author)
            ("auth5", "paper5"),  # Eve -> RL
        ]
        
        for auth_id, paper_id in authorships:
            conn.execute("""
                MATCH (a:Author {author_id: $auth_id}), 
                      (p:Paper {paper_id: $paper_id})
                CREATE (a)-[:WROTE]->(p)
            """, {"auth_id": auth_id, "paper_id": paper_id})
        print(f"    ✓ Created {len(authorships)} authorship links")
        
        # Citation relationships
        print("  Creating CITES relationships...")
        citations = [
            ("paper2", "paper1"),  # Deep Learning cites ML Fundamentals
            ("paper3", "paper1"),  # NLP cites ML Fundamentals
            ("paper4", "paper1"),  # GNN cites ML Fundamentals
            ("paper4", "paper2"),  # GNN cites Deep Learning
            ("paper5", "paper2"),  # RL cites Deep Learning
        ]
        
        for citing, cited in citations:
            conn.execute("""
                MATCH (p1:Paper {paper_id: $citing}),
                      (p2:Paper {paper_id: $cited})
                CREATE (p1)-[:CITES]->(p2)
            """, {"citing": citing, "cited": cited})
        print(f"    ✓ Created {len(citations)} citation links")
        
        # ========================================
        # VERIFICATION
        # ========================================
        print("\n" + "="*50)
        print("VERIFICATION")
        print("="*50)
        
        result = conn.execute("MATCH (a:Author) RETURN COUNT(a) AS count")
        author_count = result.get_next()[0]
        print(f"✓ Authors in database: {author_count}")
        
        result = conn.execute("MATCH (p:Paper) RETURN COUNT(p) AS count")
        paper_count = result.get_next()[0]
        print(f"✓ Papers in database: {paper_count}")
        
        result = conn.execute("MATCH ()-[r:WROTE]->() RETURN COUNT(r) AS count")
        wrote_count = result.get_next()[0]
        print(f"✓ WROTE relationships: {wrote_count}")
        
        result = conn.execute("MATCH ()-[r:CITES]->() RETURN COUNT(r) AS count")
        cites_count = result.get_next()[0]
        print(f"✓ CITES relationships: {cites_count}")
        
        print("\n" + "="*50)
        print("SUCCESS! Test database created at:", TEST_DB_PATH)
        print("="*50)
        print("\nYou can now run your tests with:")
        print(f"  export DB_PATH={TEST_DB_PATH}")
        print(f"  pytest tests/ -v")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        conn.close()
    
    return True


def verify_database():
    """Verify the test database was set up correctly."""
    if not os.path.exists(TEST_DB_PATH):
        print(f"❌ Test database not found at {TEST_DB_PATH}")
        return False
    
    try:
        db = Database(TEST_DB_PATH)
        conn = Connection(db)
        
        # Quick verification query
        result = conn.execute("""
            MATCH (a:Author)-[:WROTE]->(p:Paper)
            RETURN a.name, p.title
            LIMIT 3
        """)
        
        print("\nSample data:")
        for row in result:
            print(f"  {row[0]} wrote '{row[1]}'")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("TEST DATABASE SETUP")
    print("="*50)
    
    success = setup_test_database()
    
    if success:
        print("\n" + "="*50)
        print("Verifying database...")
        print("="*50)
        verify_database()
        sys.exit(0)
    else:
        sys.exit(1)