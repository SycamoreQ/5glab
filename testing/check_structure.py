# check_s2_structure.py
import json

with open('data/s2_arxiv/s2-papers-000.jsonl_arxiv.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Check first 10 papers
            break
        
        paper = json.loads(line)
        
        print(f"\nPaper {i+1}:")
        print(f"  corpusid: {paper.get('corpusid')}")
        print(f"  authors: {len(paper.get('authors', []))} authors")
        print(f"  references: {len(paper.get('references', []))} refs")
        
        # Show structure
        if paper.get('authors'):
            print(f"  author sample: {paper['authors'][0]}")
        if paper.get('references'):
            print(f"  ref sample: {paper['references'][0]}")
