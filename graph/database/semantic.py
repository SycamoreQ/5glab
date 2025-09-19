import kuzu
from typing import List , Optional , Dict , Any 

db = kuzu.Database("academic_paper")
conn = kuzu.Connection(db)



class Paper: 

    def create_node(): 

        conn.execute(""
                     CREATE NODE TABLE Paper () 
        "")