import pickle

with open('pruned_graph_1M.pkl', 'rb') as f:
    graph_data = pickle.load(f)

all_ids = set()
for cite in graph_data['citations']:
    all_ids.add(cite['source'])
    all_ids.add(cite['target'])

print(f"Unique papers in citation graph: {len(all_ids):,}")