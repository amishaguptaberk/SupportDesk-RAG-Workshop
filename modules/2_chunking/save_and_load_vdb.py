import json
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

_here = os.path.dirname(os.path.abspath(__file__))
_chroma_dir = os.path.join(_here, 'my_chroma_db')

# Load data
with open(os.path.join(_here, '../../data/synthetic_tickets.json'), 'r') as f:
    tickets = json.load(f)

documents = [
    Document(
        page_content=f"{t['title']}. {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Step 1: Build and save (Chroma persists with persist_directory)
print("Building vector store...")
store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection",
    persist_directory=_chroma_dir
)
print(f"✓ Saved to {_chroma_dir}")

# Step 2: Load it back
print("\nLoading vector store...")
loaded_store = Chroma(
    persist_directory=_chroma_dir,
    embedding_function=embeddings,
    collection_name="my_collection"
)
print("✓ Loaded from disk")

# Step 3: Verify it works
query = "login problem"
results = loaded_store.similarity_search(query, k=3)
print(f"\nSearch results for '{query}':")
for doc in results:
    print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")