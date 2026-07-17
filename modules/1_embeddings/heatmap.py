import json
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load first 10 tickets
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, '../../data/synthetic_tickets.json'), 'r') as f:
    tickets = json.load(f)[:10]

# Generate embeddings
texts = [t['title'] for t in tickets]
response = client.embeddings.create(input=texts, model='text-embedding-3-small')
embeddings = np.array([data.embedding for data in response.data])

# Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Create heatmap
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(10), [t['ticket_id'] for t in tickets], rotation=45, ha='right')
plt.yticks(range(10), [t['ticket_id'] for t in tickets])
plt.title('Ticket Similarity Matrix')
plt.tight_layout()
plt.savefig(os.path.join(_here, 'similarity_heatmap.png'))
plt.show()
print("✓ Saved as similarity_heatmap.png")
