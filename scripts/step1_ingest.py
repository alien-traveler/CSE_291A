# scripts/step1_ingest.py
from datasets import load_dataset
import pandas as pd
import re, json, pathlib

OUT = pathlib.Path("data"); (OUT/"raw").mkdir(parents=True, exist_ok=True); (OUT/"logs").mkdir(parents=True, exist_ok=True)

# fast bootstrap; you can swap to Kaggle/arXiv API later
ds = load_dataset("jamescalam/ai-arxiv")["train"]
df = pd.DataFrame(ds)

keep = ["id", "title", "summary", "source", "authors", "categories", "published", "updated"]
df = df[keep].rename(columns={"id":"arxiv_id","summary":"abstract"})
df["title"]    = df["title"].fillna("").str.strip()
df["abstract"] = df["abstract"].fillna("").str.replace(r"\s+"," ",regex=True).str.strip()
df = df[df["abstract"].str.len()>0]

# normalize categories and primary_category
def primary(cat):
    if not cat: return None
    # pick first cs.* if present, else first
    cs = [c for c in cat if str(c).startswith("cs.")]
    return (cs[0] if cs else cat[0]) if isinstance(cat, list) else cat
df["categories"] = df["categories"].apply(lambda x: x if isinstance(x, list) else re.split(r"\s+", str(x).strip()))
df["primary_category"] = df["categories"].apply(primary)

# dedupe
df = df.drop_duplicates(subset=["arxiv_id"]).reset_index(drop=True)

# save outputs
df.to_parquet("data/paper_meta.parquet", index=False)
df.to_json("data/raw/arxiv_meta_snapshot.jsonl", orient="records", lines=True)

# simple report
report = {
  "num_rows": len(df),
  "date_min": str(df["published"].min()),
  "date_max": str(df["published"].max()),
  "top_categories": df["primary_category"].value_counts().head(10).to_dict()
}
(OUT/"logs"/"ingest_report.json").write_text(json.dumps(report, indent=2))
print("✅ step1 done:", report)
