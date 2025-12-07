# scripts/step3_embed_bge.py
import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer

# load chunks and metadata
chunks = pd.read_parquet("data/paper_chunks.parquet")
meta = pd.read_parquet("data/paper_meta.parquet").set_index("arxiv_id")

# BGE tip: add "passage: " to corpus texts and prepend metadata
corpus_texts = []
for _, row in chunks.iterrows():
    m = meta.loc[row["arxiv_id"]]
    title = m.get("title", "") or row["arxiv_id"]
    authors = m.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]
    author_str = ", ".join(authors) if len(authors) else "Unknown authors"
    enriched = f"Title: {title}\nAuthors: {author_str}\n\n{row['chunk_text']}"
    corpus_texts.append("passage: " + enriched)

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
