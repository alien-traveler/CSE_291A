# scripts/search_bge.py
import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

index = faiss.read_index("index/arxiv_abs_bge_ip.faiss")
lookup = pd.read_parquet("index/lookup.parquet")
meta   = pd.read_parquet("data/paper_meta.parquet").set_index("arxiv_id")

enc = SentenceTransformer("BAAI/bge-large-en-v1.5")

def search(query, k=10):
    q_text = "query: " + query
    q_vec = enc.encode([q_text], normalize_embeddings=True)
    D, I = index.search(np.asarray(q_vec, dtype="float32"), k)
    rows = []
    for score, idx in zip(D[0], I[0]):
        row = lookup.iloc[idx]
        m = meta.loc[row["arxiv_id"]]
        rows.append({
            "score": float(score),
            "arxiv_id": row["arxiv_id"],
            "title": m["title"],
            "abstract": m["abstract"],
            "primary_category": m.get("primary_category", None),
            "published": m.get("published", None),
            "source_url": m.get("source_url", f"https://arxiv.org/abs/{row['arxiv_id']}")
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    hits = search("test-time compute scaling for LLMs", k=5)
    print(hits[["score","arxiv_id","title","published"]])
