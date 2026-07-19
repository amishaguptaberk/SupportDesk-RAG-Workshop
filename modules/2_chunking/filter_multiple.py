import json
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

_here = os.path.dirname(os.path.abspath(__file__))

# Load and create documents
with open(os.path.join(_here, '../../data/synthetic_tickets.json'), 'r') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    doc = Document(
        page_content=f"{ticket['title']}. {ticket['description']}",
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

# Create vector store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
store = Chroma.from_documents(documents, embeddings, collection_name="exercise7")

# Search with combined filter
query = "system not working"

# Chroma uses $and for multiple conditions
combined_filter = {"$and": [{"category": "Authentication"}, {"priority": "High"}]}

results = store.similarity_search(query, k=3, filter=combined_filter)

print(f"Query: '{query}'")
print(f"Filter: High priority + Authentication category")
print(f"\nResults ({len(results)}):")
for doc in results:
    print(f"  [{doc.metadata['priority']}] [{doc.metadata['category']}] {doc.metadata['ticket_id']}")