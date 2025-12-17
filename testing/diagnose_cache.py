from collections import Counter
from RL.env import AdvancedGraphTraversalEnv

env = AdvancedGraphTraversalEnv(store)
c = Counter()
for src, edges in env.training_edge_cache.items():
    c.update([et for et, _ in edges])
print(c)
