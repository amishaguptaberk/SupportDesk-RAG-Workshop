import json
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

# Load data
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, '../../data/synthetic_tickets.json'), 'r') as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

def search_with_category(query, category_filter=None, top_k=5):
    """Search tickets, optionally filtering by category"""
    # Get query embedding
    response = client.embeddings.create(input=[query], model=model)
    query_emb = np.array([response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get results with category filter
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]
        
        # FILL IN THIS LINE: Skip if category doesn't match filter
        # Hint: if category_filter is set AND ticket category doesn't match, skip
        if category_filter and ticket['category'] != category_filter:
            continue
        
        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    
    return results

# Test it
print("All categories:")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category:")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")