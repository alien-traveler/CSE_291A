# -*- coding: utf-8 -*-
"""
Convert full README files to RAG JSONL (no chunking)
----------------------------------------------------
Each README is embedded as a single text unit.
"""
import os
import json
from tqdm import tqdm

def readme_to_jsonl(readme_dir="github_readmes", save_path="rag_readme_full.jsonl"):
    files = [f for f in os.listdir(readme_dir) if f.endswith(".md")]

    all_data = []
    for fname in tqdm(files, desc="Processing READMEs"):
        path = os.path.join(readme_dir, fname)
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
            if len(text.strip()) < 100:
                continue  # skip trivial readmes

            all_data.append({
                "id": fname,
                "paper_id": fname.replace(".md", ""),
                "text": text,
                "metadata": {
                    "source": path,
                    "filename": fname,
                    "type": "readme_full"
                }
            })
        except Exception as e:
            print(f"⚠️ Failed to process {fname}: {e}")

    with open(save_path, "w", encoding="utf-8") as f:
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved {len(all_data)} full README entries to {save_path}")


if __name__ == "__main__":
    readme_to_jsonl("github_readmes", "rag_readme.jsonl")

