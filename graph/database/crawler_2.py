import ijson
from neo4j import GraphDatabase
from tqdm import tqdm

DB_URI = "neo4j://localhost:7687"
DB_AUTH = ("neo4j", "diam0ndman@3") 
driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)
BATCH_SIZE = 25000


def index_schema():
    with driver.session(database="researchdb") as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)")
    print("Indexes ensured.")

def parse_paper(element):
    paper_id = str(element['id'])
    doi = (element.get('doi', '') or '').lower().replace('https://doi.org/', '').strip()
    node_id = doi if doi else f"KAGGLE_{paper_id}"

    props = {
        "id": node_id,
        "kaggle_id": paper_id,
        "doi": doi,
        "title": element.get('title', ''),
        "year": element.get('year'),
        "publisher": element.get('publisher', ''),
        "venue": element.get('venue', {}).get('raw', ''),
        "keyword": ';'.join([k.get('name', '') for k in element.get('fos', []) if 'name' in k]),
    }
    # Authors: [{name,id,org}]
    authors = element.get('authors', [])
    author_l = [
        {
            'id': a.get('id', f'NOID_{a.get("name","")}' if a.get('name') else None),
            'name': a.get('name', ''),
            'org': a.get('org', '')
        }
        for a in authors if 'name' in a
    ]
    cited_ids = [str(cid) for cid in element.get('references', [])] if 'references' in element else []
    return node_id, props, author_l, cited_ids

def load_kaggle_json(json_path):
    index_schema()
    papers = []
    authorships = []
    author_nodes = {}
    cites_edges = []

    print("Parsing and batch loading...")
    with open(json_path, "rb") as f:
        objects = ijson.items(f, "item")
        batch = []
        for elem in tqdm(objects):
            node_id, props, author_l, cited_ids = parse_paper(elem)
            papers.append({"id": node_id, "props": props})
            # Prepare authors and authorshops
            for a in author_l:
                author_nodes[a['id']] = a
                authorships.append({"author_id": a['id'], "paper_id": node_id})
            # Prepare citations
            for toid in cited_ids:
                cites_edges.append({"from_id": node_id, "to_id": toid})

            # Batching
            if len(papers) >= BATCH_SIZE:
                commit_to_neo4j(papers, author_nodes, authorships, cites_edges)
                papers.clear()
                authorships.clear()
                author_nodes.clear()
                cites_edges.clear()

        if papers:
            commit_to_neo4j(papers, author_nodes, authorships, cites_edges)

    print("Finished bulk import of Kaggle dataset.")

def commit_to_neo4j(papers, author_nodes, authorships, cites_edges):
    with driver.session(database="researchdb") as session:
        if papers:
            session.run("""
                UNWIND $batch AS row
                MERGE (p:Paper {id: row.id})
                SET p += row.props
            """, batch=papers)
        if author_nodes:
            session.run("""
                UNWIND $batch AS row
                MERGE (a:Author {id: row.id})
                SET a.name = row.name, a.org = row.org
            """, batch=list(author_nodes.values()))
        if authorships:
            session.run("""
                UNWIND $batch AS row
                MATCH (a:Author {id: row.author_id})
                MATCH (p:Paper {id: row.paper_id})
                MERGE (a)-[:WROTE]->(p)
            """, batch=authorships)
        if cites_edges:
            session.run("""
                UNWIND $batch AS row
                MATCH (s:Paper {id: row.from_id})
                MATCH (t:Paper {id: row.to_id})
                MERGE (s)-[:CITES]->(t)
            """, batch=cites_edges)


def build_node_id(x, doi_set=None):
    """
    Convert a raw id or doi to the canonical id used for node insertion.
    If doi_set is provided and x matches a DOI, return as is; else, "KAGGLE_{id}"
    """
    if isinstance(x, str) and "/" in x:
        return x.lower().strip()
    return f"KAGGLE_{x}"

def load_cites_edges_only(json_path, batch_size=25000):
    """
    Second pass: only add CITES edges for already-inserted papers.
    Uses the same canonicalization for node IDs.
    """
    print("Starting CITES edge insertion pass...")
    cites_edges = []

    with driver.session(database="researchdb") as session:
        # Gather all existing Paper ids into a set for fast lookup
        print("Indexing all Paper nodes in Neo4j...")
        node_ids = set()
        result = session.run("MATCH (p:Paper) RETURN p.id")
        for rec in result:
            node_ids.add(rec["p.id"])
        print(f"Indexed {len(node_ids)} paper nodes.")

    # Now parse JSON and build CITES edges for valid (src, tgt) pairs
    with open(json_path, "rb") as f:
        objs = ijson.items(f, "item")
        batch = []
        for elem in tqdm(objs, desc="Building CITES edges"):
            src_id = build_node_id(elem.get('doi', '') or elem['id'])
            if src_id not in node_ids:
                continue
            # Note: Handle int vs. string reference ids robustly
            for ref in elem.get('references', []):
                tgt_id = build_node_id(ref)
                if tgt_id in node_ids:
                    batch.append({"from_id": src_id, "to_id": tgt_id})
            if len(batch) >= batch_size:
                commit_cites_edges(batch)
                batch.clear()
        if batch:
            commit_cites_edges(batch)

    print("CITES-edge loading complete.")

def commit_cites_edges(cites_edges):
    with driver.session(database="researchdb") as session:
        session.run("""
            UNWIND $batch AS row
            MATCH (src:Paper {id: row.from_id})
            MATCH (dst:Paper {id: row.to_id})
            MERGE (src)-[:CITES]->(dst)
        """, batch=cites_edges)
