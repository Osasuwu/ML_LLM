"""Test query with different top_k values to see which chunks are retrieved."""
from corporate_llm import CorporateLLM

# Initialize system
llm = CorporateLLM()

# Test query
question = "Процент повторного использования фичей"

print("="*80)
print(f"TESTING QUERY: {question}")
print("="*80)

# Try with different top_k values
for k in [3, 5, 8, 10]:
    print(f"\n{'='*80}")
    print(f"TOP-{k} RESULTS:")
    print(f"{'='*80}")
    
    context_chunks = llm.vector_store.search(question, top_k=k)
    
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk['metadata'].get('filename', 'Unknown')
        distance = chunk.get('distance', 'N/A')
        
        # Check if this chunk contains the answer
        contains_answer = 'повторного использования фичей' in chunk['content'].lower()
        marker = " ✓ CONTAINS ANSWER!" if contains_answer else ""
        
        print(f"{i}. {source} (distance: {distance:.4f}){marker}")
        
        if contains_answer:
            # Show the relevant part
            content_lower = chunk['content'].lower()
            idx = content_lower.find('повторного использования фичей')
            start = max(0, idx - 100)
            end = min(len(chunk['content']), idx + 200)
            print(f"   Context: ...{chunk['content'][start:end]}...")
    
    print()
