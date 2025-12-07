import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What are the main contributions of vision-language models in 2025?")
    parser.add_argument("--source", choices=["paper", "readme"], default="paper")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model = SentenceTransformer("BAAI/bge-m3", cache_folder="./model")
    query_vec = model.encode([args.query], normalize_embeddings=True)

    if args.source == "paper":
        index_path = "rag_paper_index.faiss"
        data_path = "rag_paper.jsonl"
    else:
        index_path = "rag_readme_index.faiss"
        data_path = "rag_readme.jsonl"

    index = faiss.read_index(index_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    D, I = index.search(query_vec.astype(np.float32), k=args.k)
    print(f"Top {args.k} hits from {args.source} index:")
    for rank, idx in enumerate(I[0]):
        snippet = data[idx]["text"][:500].replace("\n", " ")
        print(f"\n[{rank+1}] score={D[0][rank]:.4f} ... {snippet}...")


if __name__ == "__main__":
    main()
