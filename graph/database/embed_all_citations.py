# embed_all_citations.py
import pickle
from utils.batchencoder import BatchEncoder

def main():
    print("="*80)
    print("STEP 1: Finding missing papers")
    print("="*80)
    
    print("\nLoading graph...")
    with open('pruned_graph_1M.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    citations = graph_data['citations']
    papers = graph_data['papers']

    # Get ALL paper IDs in citations
    all_citation_ids = set()
    for cite in citations:
        all_citation_ids.add(str(cite['source']))
        all_citation_ids.add(str(cite['target']))

    print(f"✓ Unique papers in citations: {len(all_citation_ids):,}")

    # Check current coverage
    print("\nLoading existing embeddings...")
    with open('training_cache/embeddings_1M.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    embedded_ids = {str(k) for k in embeddings.keys()}
    missing_ids = all_citation_ids - embedded_ids

    print(f"✓ Already embedded: {len(embedded_ids):,}")
    print(f"✓ Missing embeddings: {len(missing_ids):,}")

    # Get paper objects for missing IDs
    paper_dict = {str(p['paperId']): p for p in papers}
    missing_papers = [paper_dict[pid] for pid in missing_ids if pid in paper_dict]

    print(f"✓ Papers available to embed: {len(missing_papers):,}")

    if not missing_papers:
        print("\n🎉 All papers already embedded! Nothing to do.")
        return

    # Save for reference
    with open('training_cache/papers_to_embed.pkl', 'wb') as f:
        pickle.dump(missing_papers, f)
    print("✓ Saved papers_to_embed.pkl")

    # Estimate time
    estimated_minutes = len(missing_papers) / 256 / 60
    print(f"\n⏱️  Estimated time: {estimated_minutes:.1f} minutes at 256 batch size")
    print("   (approximately {:.1f} hours)".format(estimated_minutes / 60))
    
    # Confirmation
    response = input("\n🚀 Start embedding now? This will take a while. (y/n): ")
    if response.lower() != 'y':
        print("Aborted. Run this script again when ready.")
        return

    print("\n" + "="*80)
    print("STEP 2: Embedding missing papers")
    print("="*80)
    print("☕ Get coffee - this will take 2-3 hours...\n")

    encoder = BatchEncoder(
        model_name='all-MiniLM-L6-v2',
        batch_size=256,
        cache_file='training_cache/embeddings_1M.pkl'  # appends to existing
    )

    # Embed with progress tracking
    encoder.precompute_paper_embeddings(missing_papers, force=False)

    print("\n" + "="*80)
    print("✅ EMBEDDING COMPLETE!")
    print("="*80)
    print(f"Total embeddings: {len(encoder.cache):,}")
    print(f"New embeddings: {len(missing_papers):,}")
    print("\n📋 Next steps:")
    print("  1. Run: python -m graph.database.cache_once")
    print("  2. Run: python -m RL.train_rl")

if __name__ == '__main__':
    main()
