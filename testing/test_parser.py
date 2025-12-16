# testing/test_parser.py
from model.llm.parser.dspy_parser import DSPyHierarchicalParser

def test():
    parser = DSPyHierarchicalParser(model="llama3.2", optimize=False)
    
    query = "Get me authors who wrote papers in 'IEEJ Transactions' in the field of Physics with more than 50 citations"
    
    print(f"Query: {query}\n")
    intent = parser.parse(query)
    
    print(f"Target Entity: {intent.target_entity}")
    print(f"Operation: {intent.operation}")
    print(f"Semantic: {intent.semantic}")
    print(f"Constraints ({len(intent.constraints)}):")
    for c in intent.constraints:
        print(f"  • {c}")
    
    # Expected output:
    # Target Entity: authors
    # Operation: find
    # Semantic: Physics
    # Constraints (3):
    #   • venue contains IEEJ Transactions
    #   • field_of_study contains Physics
    #   • citation_count greater_than 50

if __name__ == "__main__":
    test()
