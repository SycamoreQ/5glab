"""Test paper-specific query parsing."""

from model.llm.parser.dspy_parser import OptimQueryParser


def test_paper_queries():
    parser = OptimQueryParser()
    
    test_queries = [
        "Get me the citations of this paper titled Attention Is All You Need",
        "what papers does the BERT paper cite",
        "show me papers similar to AlexNet",
        "who are the authors of ImageNet Classification paper",
        "find papers that cite ResNet and are published in CVPR",
        "what are the second-order citations of GPT-3",
        "references of Transformer paper from 2020 onwards",
    ]
    
    print("\n" + "="*80)
    print("PAPER-SPECIFIC QUERY PARSING TEST")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        facets = parser.parse(query)
        
        print(f"  Paper title: {facets['paper_title']}")
        print(f"  Paper-centric: {facets['paper_search_mode']}")
        print(f"  Operation: {facets['paper_operation']}")
        print(f"  Relation focus: {facets['relation_focus']}")
        print(f"  Hop depth: {facets['hop_depth']}")
        print(f"  Semantic: {facets['semantic']}")
        print(f"  Temporal: {facets['temporal']}")
        print(f"  Venue: {facets['venue']}")


if __name__ == "__main__":
    test_paper_queries()
