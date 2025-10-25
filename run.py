# -*- coding: utf-8 -*-
"""
RAG Pipeline (Local vLLM + BGE-Reranker version)
------------------------------------------------
Features:
- FAISS vector retrieval (BGE embedding)
- CrossEncoder reranker for precision
- Local vLLM model generation via OpenAI-compatible API
"""

import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


# ====================== 全局配置 ======================
EMBED_MODEL = "BAAI/bge-m3"                   # 向量模型
RERANK_MODEL = "BAAI/bge-reranker-large"      # 重排序模型
INDEX_PATH = "rag_index.faiss"
EMBED_PATH = "rag_embeddings.npy"
DATA_PATH = "rag_data.jsonl"

# ✅ 本地 vLLM 服务配置
VLLM_BASE_URL = "http://localhost:8082/v1"
VLLM_MODEL = "Qwen/Qwen3-0.6B"       # 模型名与启动命令保持一致


# ====================== 初始化模型 ======================
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL, cache_folder="./model")

print("🔹 Loading FAISS index and embeddings...")
index = faiss.read_index(INDEX_PATH)
embeddings = np.load(EMBED_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
print(f"✅ Loaded {len(data)} chunks from {DATA_PATH}")

print("🔹 Loading reranker model (this may take a while)...")
reranker = CrossEncoder(RERANK_MODEL, cache_folder="./model")

# ✅ 初始化本地 vLLM 客户端
client = OpenAI(
    base_url=VLLM_BASE_URL,  # vLLM 启动时的地址
    api_key="EMPTY"           # 任意字符串即可（vLLM 不校验）
)


# ====================== 阶段 1：检索 ======================
def retrieve(query: str, top_k: int = 20):
    """
    Step 1: 从 FAISS 检索 top_k 相关文段（粗检索）
    """
    query_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec.astype(np.float32), top_k)
    results = [data[i] for i in I[0]]
    return results


# ====================== 阶段 2：重排序 ======================
def rerank_results(query: str, retrieved_chunks, top_k: int = 5):
    """
    Step 2: 使用 CrossEncoder 对 FAISS 检索结果进行二次排序（精排）
    """
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    reranked_chunks = [r[0] for r in reranked[:top_k]]

    print(f"🔍 Reranked {len(retrieved_chunks)} → top {top_k}")
    return reranked_chunks


# ====================== 阶段 3：生成 ======================
def rag_generate(query: str, top_k: int = 5):
    """
    Step 3: 基于重排后的文段生成回答（使用本地 vLLM）
    """
    # ① FAISS 检索
    retrieved_chunks = retrieve(query, top_k=20)
    # ② Reranker 精排
    reranked_chunks = rerank_results(query, retrieved_chunks, top_k=top_k)

    # 拼接上下文
    context = "\n\n".join([r["text"] for r in reranked_chunks])

    # 生成 prompt
    prompt = f"""
You are a helpful academic assistant.
Use the provided context to answer the question as accurately as possible.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {query}
Answer:
"""

    # ✅ 使用本地 vLLM 生成回答
    completion = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert assistant for summarizing and analyzing research papers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = completion.choices[0].message.content.strip()
    return answer, reranked_chunks


# ====================== 阶段 4：交互界面 ======================
if __name__ == "__main__":
    # print("\n🚀 Academic RAG Assistant (Local vLLM version) ready!\n")
    # while True:
    #     query = input("🔍 Enter your question (or 'exit' to quit): ").strip()
    #     if query.lower() in ["exit", "quit"]:
    #         break

    #     answer, refs = rag_generate(query, top_k=5)

    #     print("\n💡 Answer:\n", answer)
    #     print("\n📚 Top References:")
    #     for r in refs:
    #         meta = r["metadata"]
    #         print(f" - {meta.get('filename')} (chunk {meta.get('chunk_id')})")
    #     print("\n" + "-" * 60 + "\n")
    
    # ✅ 直接在代码中设置查询
    query = "What are the main contributions of vision-language models in 2025?"
    
    print(f"🔍 Query: {query}\n")
    
    answer, refs = rag_generate(query, top_k=5)

    print("\n💡 Answer:\n", answer)
    print("\n📚 Top References:")
    for r in refs:
        meta = r["metadata"]
        print(f" - {meta.get('filename')} (chunk {meta.get('chunk_id')})")
    print("\n" + "-" * 60 + "\n")


