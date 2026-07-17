import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

# Method 1: SLOW - One API call per text
print("Method 1: Single API calls...")
start = time.time()
for text in texts:
    response = client.embeddings.create(input=[text], model=model)
time_slow = time.time() - start
print(f"  Time: {time_slow:.2f} seconds")

# Method 2: FAST - One API call for all texts
print("\nMethod 2: Batch API call...")
start = time.time()
response = client.embeddings.create(input=texts, model=model)
time_fast = time.time() - start
print(f"  Time: {time_fast:.2f} seconds")

# Compare
print(f"\n✓ Batch is {time_slow/time_fast:.1f}x faster!")
print(f"  Always batch your embeddings in production!")