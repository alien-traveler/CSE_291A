import json
import os
from typing import Callable, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_arxiv_meta(meta_path: str) -> Dict[str, Dict]:
    """Map arxiv_id -> {title, authors} (best-effort)."""
    if not os.path.exists(meta_path):
        return {}
    meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta[row.get("id")] = {
                "title": row.get("title"),
                "authors": row.get("authors"),
            }
    return meta


def load_readme_meta(meta_path: str) -> Dict[str, Dict]:
    """Map filename -> {title} using crawl metadata when available."""
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        filename = os.path.basename(row.get("readme_path", ""))
        if filename:
            out[filename] = {
                "title": row.get("repo") or filename.replace("_README.md", ""),
            }
    return out


def prepend_paper_metadata(record: Dict, arxiv_meta: Dict[str, Dict]) -> str:
    paper_id = record.get("paper_id") or record.get("metadata", {}).get("filename", "")
    meta = arxiv_meta.get(paper_id, {})
    title = meta.get("title") or paper_id or "Unknown title"
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author_str = ", ".join(authors) if authors else "Unknown authors"
    return f"Title: {title}\nAuthors: {author_str}\n\n{record['text']}"


def prepend_readme_metadata(record: Dict, readme_meta: Dict[str, Dict]) -> str:
    filename = record.get("metadata", {}).get("filename", "")
    meta = readme_meta.get(filename, {})
    title = meta.get("title") or filename or record.get("paper_id") or "README"
    return f"Title: {title}\n\n{record['text']}"


def build_index(
    records: List[Dict],
    model: SentenceTransformer,
    prefix_fn: Callable[[Dict], str],
    embeddings_path: str,
    index_path: str,
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    texts = [prefix_fn(r) for r in records]

    print(f"⚙️ Generating embeddings for {len(texts)} chunks -> {os.path.basename(index_path)}")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(embeddings_path, embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    print(f"✅ Saved embeddings to {embeddings_path}")
    print(f"✅ Saved FAISS index to {index_path}")
    return index, embeddings


def main():
    model_name = "BAAI/bge-m3"
    print(f"🔹 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder="./model")

    # Papers (data/)
    paper_records = load_jsonl("rag_paper.jsonl")
    arxiv_meta = load_arxiv_meta("arxiv_llm_vlm_2025.jsonl")
    build_index(
        paper_records,
        model,
        lambda r: prepend_paper_metadata(r, arxiv_meta),
        embeddings_path="rag_paper_embeddings.npy",
        index_path="rag_paper_index.faiss",
    )

    # GitHub READMEs (github_readmes/)
    readme_records = load_jsonl("rag_readme.jsonl")
    readme_meta = load_readme_meta(os.path.join("github_readmes", "readme_metadata.json"))
    build_index(
        readme_records,
        model,
        lambda r: prepend_readme_metadata(r, readme_meta),
        embeddings_path="rag_readme_embeddings.npy",
        index_path="rag_readme_index.faiss",
    )


if __name__ == "__main__":
    main()
