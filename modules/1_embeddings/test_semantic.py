import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

# These mean the SAME thing but use DIFFERENT words
texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # DIFFERENT topic
]

# Generate embeddings
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

# Calculate all pairwise similarities
similarity_matrix = cosine_similarity(embeddings)

# Print results
print("Similarity Matrix:")
print("-" * 50)
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:  # Only print upper triangle
            sim = similarity_matrix[i][j]
            print(f"{sim:.3f}  '{text1[:30]}...' vs '{text2[:30]}...'")