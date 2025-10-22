# scripts/step3_embed_bge.py
import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer

# load chunks
chunks = pd.read_parquet("data/paper_chunks.parquet")

# BGE tip: add "passage: " to corpus texts
corpus_texts = ["passage: " + t for t in chunks["chunk_text"].tolist()]

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
emb = model.encode(corpus_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

# FAISS IndexFlatIP works with normalized vectors = cosine similarity
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb.astype("float32"))

# persist
faiss.write_index(index, "index/arxiv_abs_bge_ip.faiss")
chunks[["chunk_id","arxiv_id","section","order"]].to_parquet("index/lookup.parquet", index=False)
print("✅ step3 done: index size =", index.ntotal)
