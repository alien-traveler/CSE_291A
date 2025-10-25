import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss

query = "What are the main contributions of vision-language models in 2025?"
model = SentenceTransformer("BAAI/bge-m3",cache_folder='./model')
query_vec = model.encode([query], normalize_embeddings=True)

index = faiss.read_index("rag_index.faiss")
embeddings = np.load("rag_embeddings.npy")
with open("rag_data.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

D, I = index.search(query_vec.astype(np.float32), k=5)
for rank, idx in enumerate(I[0]):
    print(f"\n[{rank+1}] {data[idx]['text'][:500]}...")