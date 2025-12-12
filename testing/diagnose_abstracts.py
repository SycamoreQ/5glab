# check_abstract.py
import gzip
import json
import os

download_dir = "/Users/kaushikmuthukumar/Downloads/s2_datasets"
gz_files = [f for f in os.listdir(download_dir) if f.endswith('.gz')]

if gz_files:
    filepath = os.path.join(download_dir, gz_files[0])
    print(f"Checking file: {gz_files[0]}\n")
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        # Check first 5 papers
        for i in range(5):
            line = f.readline()
            if not line:
                break
            
            paper = json.loads(line)
            
            print(f"=== Paper {i+1} ===")
            print(f"Title: {paper.get('title', 'N/A')[:80]}")
            print(f"Corpus ID: {paper.get('corpusid')}")
            
            abstract = paper.get('abstract')
            if abstract:
                print(f"Abstract: {abstract[:100]}...")
            else:
                print("Abstract: MISSING")
            
            print()
else:
    print("No .gz files found!")
