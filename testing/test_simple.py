"""
Minimal test to verify database setup works.
Run this first to debug the path issue.

Usage:
    pytest test_simple.py -v -s
"""

import pytest
import os
import tempfile
import shutil
from kuzu import Database, Connection


def test_kuzu_database_creation():
    """Test that we can create a KuzuDB database correctly."""
    
    # Create parent directory
    temp_parent = tempfile.mkdtemp(prefix="test_kuzu_parent_")
    
    # DB path is INSIDE the parent, but doesn't exist yet
    db_path = os.path.join(temp_parent, "my_test_db")
    
    print(f"\nParent dir: {temp_parent}")
    print(f"DB path: {db_path}")
    print(f"DB path exists before: {os.path.exists(db_path)}")
    
    try:
        # KuzuDB will CREATE this directory
        db = Database(db_path)
        conn = Connection(db)
        
        print(f"DB path exists after: {os.path.exists(db_path)}")
        
        # Create a simple table to verify it works
        conn.execute("""
            CREATE NODE TABLE Person (
                id STRING PRIMARY KEY,
                name STRING
            )
        """)
        
        conn.execute(
            "CREATE (p:Person {id: 'p1', name: 'Test'})",
            {}
        )
        
        result = conn.execute("MATCH (p:Person) RETURN p.name")
        row = result.get_next()
        
        assert row[0] == "Test"
        print("✓ Database works correctly!")
        
        conn.close()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_parent, ignore_errors=True)
        print(f"Cleaned up {temp_parent}")


if __name__ == "__main__":
    test_kuzu_database_creation()
    print("\n✓ All tests passed!")