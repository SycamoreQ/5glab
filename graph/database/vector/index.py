from store import * 
import logging 
from typing import Dict , Any 
import kuzu 



def create_vector_index(conn: kuzu.Connection, table_name: str, index_name: str) -> None:
    """Create vector index on the given table and column name"""
    try:
        conn.execute("INSTALL vector; LOAD vector;")
    except RuntimeError:
        print("Vector extension already installed and loaded.")
    conn.execute(
        f"""
        CALL CREATE_VECTOR_INDEX(
            '{table_name}',
            '{index_name}',
            'vector'
        );
        """
    )
    print(f"Vector index created for {table_name} table.")


