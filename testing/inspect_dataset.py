# save as inspect_dataset.py
import gzip
import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else './s2_datasets/20251205_080609_00052_fiwdp_1e7c2236-d997-4720-9dd2-ad082a414df3.gz'

print(f"Inspecting: {filepath}\n")

with gzip.open(filepath, 'rt', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        paper = json.loads(line)
        print(f"=== PAPER {i+1} ===")
        print(f"Keys: {list(paper.keys())}")
        print(f"Sample data: {json.dumps(paper, indent=2)[:500]}")
        print()
