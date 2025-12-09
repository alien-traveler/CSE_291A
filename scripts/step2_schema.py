import pandas as pd
import uuid

meta = pd.read_parquet("data/paper_meta.parquet")

chunks = pd.DataFrame({
    "chunk_id": [f"{aid}::abs" for aid in meta["arxiv_id"]],
    "arxiv_id": meta["arxiv_id"],
    "section": ["Abstract"] * len(meta),
    "chunk_text": meta["title"].fillna("") + " || " + meta["abstract"].fillna(""),
    "order": [0]*len(meta)
})
chunks.to_parquet("data/paper_chunks.parquet", index=False)

# (Optional) — preload a tiny table that maps arxiv_id→pwc/github (fill later)
links = pd.DataFrame(columns=["arxiv_id","pwc_url","github_urls"])
links.to_parquet("data/paper_links.parquet", index=False)

print("✅ step2 done:", len(chunks), "abstract-chunks")
