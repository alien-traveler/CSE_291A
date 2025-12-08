# -*- coding: utf-8 -*-
"""
RAG Retrieval Evaluation Script - Baseline vs Advanced (Rerank)
=============================================================
"""
import json
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from rag_evaluator import evaluate
from typing import Dict, List, Set

# ==================== 1. 加载模型和数据 ====================
print("🔹 Loading models and data...")

# 1.1 Bi-Encoder (用于检索)
bi_encoder = SentenceTransformer("BAAI/bge-m3", cache_folder='./model')
index = faiss.read_index("rag_index.faiss")

# 1.2 Cross-Encoder (用于重排序)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', default_activation_function=None)

# 1.3 加载数据
with open("rag_all.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open("arxiv_paper.json", "r", encoding="utf-8") as f:
    arxiv_questions = json.load(f)
with open("github_readme.json", "r", encoding="utf-8") as f:
    github_questions = json.load(f)

all_questions = arxiv_questions + github_questions
print(f"✅ Loaded {len(data)} chunks and {len(all_questions)} questions.")

# ==================== 2. 构建 BM25 索引 (仅 README) ====================
print("🔹 Building BM25 index for READMEs...")

readme_indices = []
readme_corpus = []

for idx, chunk in enumerate(data):
    meta = chunk["metadata"]
    if "readme" in meta.get("filename", "").lower() or "github" in meta.get("source", "").lower():
        readme_indices.append(idx)
        readme_corpus.append(chunk["text"])

tokenized_corpus = [doc.lower().split() for doc in readme_corpus]
bm25 = BM25Okapi(tokenized_corpus)
bm25_idx_to_global = {i: global_idx for i, global_idx in enumerate(readme_indices)}

print(f"✅ BM25 index built for {len(readme_corpus)} README chunks.")

# ==================== 3. 定义延迟统计 ====================
latency_stats = {
    "baseline": [],
    "advanced": []
}

# ==================== 4. 检索函数定义 ====================

def retrieve_baseline(query: str, top_k: int = 10) -> List[Dict]:
    """
    Baseline: 仅 Dense Retrieval (FAISS)
    """
    t_start = time.time()
    
    # Embedding
    query_vec = bi_encoder.encode([query], normalize_embeddings=True)
    
    # Search
    D, I = index.search(query_vec.astype(np.float32), top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < 0 or idx >= len(data): continue
        chunk = data[idx]
        results.append({
            "paper_id": chunk["metadata"].get("filename", "unknown"),
            "chunk_id": f"chunk_{idx}",
            "score": float(score),
            "text": chunk["text"],
            "rank": rank + 1
        })
        
    t_end = time.time()
    latency_stats["baseline"].append((t_end - t_start) * 1000)
    return results


def retrieve_advanced(query: str, top_k: int = 10) -> List[Dict]:
    """
    Advanced: Hybrid (Dense + BM25) + Rerank (Cross-Encoder)
    """
    t_start = time.time()
    
    # --- Stage 1: Hybrid Retrieval ---
    candidate_ids: Set[int] = set()
    
    # 1.1 Dense (Top-50)
    dense_k = 50
    query_vec = bi_encoder.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec.astype(np.float32), dense_k)
    for idx in I[0]:
        if idx >= 0 and idx < len(data): candidate_ids.add(int(idx))
            
    # 1.2 BM25 (Top-50)
    bm25_k = 50
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25 = np.argsort(bm25_scores)[::-1][:bm25_k]
    for bm25_idx in top_bm25:
        if bm25_scores[bm25_idx] > 0:
            candidate_ids.add(bm25_idx_to_global[bm25_idx])
            
    candidates = list(candidate_ids)
    if not candidates: return []

    # --- Stage 2: Reranking ---
    cross_inp = [[query, data[idx]["text"]] for idx in candidates]
    cross_scores = cross_encoder.predict(cross_inp)
    
    scored_candidates = list(zip(candidates, cross_scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    final_results = scored_candidates[:top_k]
    
    results = []
    for rank, (idx, score) in enumerate(final_results):
        chunk = data[idx]
        results.append({
            "paper_id": chunk["metadata"].get("filename", "unknown"),
            "chunk_id": f"chunk_{idx}",
            "score": float(score),
            "text": chunk["text"],
            "rank": rank + 1
        })
        
    t_end = time.time()
    latency_stats["advanced"].append((t_end - t_start) * 1000)
    return results

# ==================== 5. 准备评估数据 ====================
# (复用之前的 prepare_evaluation_data 函数)
def prepare_evaluation_data(questions: List[Dict]) -> tuple:
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'of', 'in', 'to', 'for', 'and', 'or', 'but', 'with', 'by', 'at',
        'from', 'as', 'on', 'this', 'that', 'these', 'those', 'it', 'its'
    }
    queries = {}
    gold_paper = {}
    gold_passage = {}
    
    for idx, q in enumerate(questions):
        if "question" not in q: continue
        qid = f"q{idx}"
        question = q["question"]
        ref_contents = q.get("reference_content", [])
        source_file = q["file"]
        
        queries[qid] = {"query": question, "key_terms": ref_contents}
        gold_paper[qid] = {}
        gold_passage[qid] = {}
        
        for chunk_idx, chunk in enumerate(data):
            meta = chunk["metadata"]
            chunk_file = meta.get("filename", "")
            chunk_id = f"chunk_{chunk_idx}"
            
            if source_file in chunk_file or chunk_file in source_file:
                if chunk_file not in gold_paper[qid]: gold_paper[qid][chunk_file] = 3
                if ref_contents:
                    chunk_text = chunk["text"].lower()
                    exact_matches = sum(1 for ref in ref_contents if ref.lower() in chunk_text)
                    if exact_matches > 0:
                        gold_passage[qid][chunk_id] = 3
                        continue
                    keyword_matches = 0
                    for ref in ref_contents:
                        keywords = [w for w in ref.lower().split() if len(w) > 3 and w not in STOPWORDS]
                        if not keywords: continue
                        chunk_tokens = set(chunk_text.split())
                        matched = sum(1 for kw in keywords if kw in chunk_tokens)
                        if matched / len(keywords) >= 0.4: keyword_matches += 1
                    if keyword_matches > 0: gold_passage[qid][chunk_id] = 2
                    else: gold_passage[qid][chunk_id] = 1
                else:
                    gold_passage[qid][chunk_id] = 2
    return queries, gold_paper, gold_passage

# ==================== 6. 运行评估 ====================
print("🔍 Preparing evaluation data...")
queries, gold_paper, gold_passage = prepare_evaluation_data(all_questions)

if len(queries) == 0:
    print("❌ No valid queries found!")
    exit(1)

# 预热
print("🔥 Warming up models...")
retrieve_baseline("warm up query", top_k=10)
retrieve_advanced("warm up query", top_k=10)
latency_stats["baseline"] = []
latency_stats["advanced"] = []

print("🔍 Running Baseline Retrieval...")
runs_baseline = {}
for qid, q_data in queries.items():
    runs_baseline[qid] = retrieve_baseline(q_data["query"], top_k=10)

print("🔍 Running Advanced Retrieval (Hybrid + Rerank)...")
runs_advanced = {}
for qid, q_data in queries.items():
    runs_advanced[qid] = retrieve_advanced(q_data["query"], top_k=10)

print(f"✅ Retrieved results for {len(queries)} queries\n")

# ==================== 7. 输出延迟对比 ====================
print("=" * 80)
print("⚡ LATENCY COMPARISON (ms)")
print("=" * 80)

def print_latency(name, values):
    if not values: return
    avg = sum(values) / len(values)
    p95 = np.percentile(values, 95)
    print(f"{name:20s} | Avg: {avg:6.2f}ms | P95: {p95:6.2f}ms")

print_latency("Baseline (Dense)", latency_stats["baseline"])
print_latency("Advanced (Rerank)", latency_stats["advanced"])
print("=" * 80 + "\n")

# ==================== 8. 评估指标对比 ====================
print("📊 Evaluating Baseline...")
results_baseline = evaluate(
    queries=queries,
    runs=runs_baseline,
    gold_paper=gold_paper,
    gold_passage=gold_passage,
    k_list=[5, 10],
    bootstrap=False # 快速评估
)

print("📊 Evaluating Advanced...")
results_advanced = evaluate(
    queries=queries,
    runs=runs_advanced,
    gold_paper=gold_paper,
    gold_passage=gold_passage,
    k_list=[5, 10],
    bootstrap=False
)

print("\n" + "=" * 80)
print("📈 PERFORMANCE COMPARISON (Baseline vs Advanced)")
print("=" * 80)

metrics_to_compare = [
    "paper_mrr@10", "paper_ndcg@5", 
    "passage_ndcg@5", "passage_p@5",
    "has_answer@5", "answer_coverage@5"
]

print(f"{'Metric':<25} | {'Baseline':<10} | {'Advanced':<10} | {'Diff':<10}")
print("-" * 65)

for metric in metrics_to_compare:
    base_val = results_baseline["macro"].get(metric, {}).get("mean", 0.0)
    adv_val = results_advanced["macro"].get(metric, {}).get("mean", 0.0)
    diff = adv_val - base_val
    
    diff_str = f"{diff:+.4f}"
    if diff > 0: diff_str = f"🟢 {diff_str}"
    elif diff < 0: diff_str = f"🔴 {diff_str}"
    else: diff_str = f"⚪ {diff_str}"
    
    print(f"{metric:<25} | {base_val:.4f}     | {adv_val:.4f}     | {diff_str}")

print("=" * 80 + "\n")

# 保存结果
output = {
    "baseline": results_baseline,
    "advanced": results_advanced
}
with open("evaluation_comparison.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"💾 Detailed results saved to: evaluation_comparison.json")
