import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

def build_rag_index(
    jsonl_path="rag_all.jsonl",
    index_path="rag_index.faiss",
    embeddings_path="rag_embeddings.npy",
    model_name="BAAI/bge-m3"
):
    # 1️⃣ 加载模型
    print(f"🔹 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name,cache_folder='./model')

    # 2️⃣ 读取文本数据
    print(f"📖 Reading chunks from {jsonl_path}")
    texts, metadata = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            metadata.append(item["metadata"])

    # 3️⃣ 生成嵌入
    print(f"⚙️ Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 4️⃣ 保存向量
    np.save(embeddings_path, embeddings)
    print(f"💾 Embeddings saved to {embeddings_path}")

    # 5️⃣ 建立 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积相似度（BGE 已经是 normalized）
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to {index_path}")

    return index, metadata


if __name__ == "__main__":
    build_rag_index()