# -*- coding: utf-8 -*-
"""
RAG Retrieval Evaluation - Partitioned Indices (Paper vs README)
==============================================================
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

# 1.1 Bi-Encoder
bi_encoder = SentenceTransformer("BAAI/bge-m3", cache_folder='./model')

# 1.2 加载两个独立索引 ✅
print("🔹 Loading partitioned indices...")
try:
    index_paper = faiss.read_index("rag_paper_index.faiss")
    index_readme = faiss.read_index("rag_readme_index.faiss")
    print(f"✅ Loaded Paper Index: {index_paper.ntotal} vectors")
    print(f"✅ Loaded README Index: {index_readme.ntotal} vectors")
except Exception as e:
    print(f"❌ Error loading indices: {e}")
    print("Did you run the new build_embedding.py?")
    exit(1)

# 1.3 Cross-Encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', default_activation_function=None)

# 1.4 加载数据 (假设 rag_all.jsonl 已经是包含元数据增强的新版本)
with open("rag_all.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 建立映射: Global Index -> Source Type
# 我们需要知道全局 data 列表中的每个 chunk 属于哪个索引，以便正确映射 ID
# 假设构建索引时是按顺序添加的，或者我们需要重新扫描一遍来建立映射
# ⚠️ 关键假设：rag_all.jsonl 的顺序没有变，且我们知道哪些是 paper 哪些是 readme
# 更稳妥的方法是根据 metadata 实时判断
paper_indices = []
readme_indices = []
for idx, chunk in enumerate(data):
    meta = chunk["metadata"]
    if "readme" in meta.get("filename", "").lower() or "github" in meta.get("source", "").lower():
        readme_indices.append(idx)
    else:
        paper_indices.append(idx)

# 建立局部索引到全局索引的映射
# 假设 build_embedding 时是先加所有 paper 再加所有 readme，或者分开加的
# 这里我们需要根据你的 build 逻辑来对应。
# 最通用的方式：假设 index_paper 里的第 i 个向量对应 paper_indices[i]
paper_local_to_global = {i: global_idx for i, global_idx in enumerate(paper_indices)}
readme_local_to_global = {i: global_idx for i, global_idx in enumerate(readme_indices)}

with open("test/arxiv_paper.json", "r", encoding="utf-8") as f:
    arxiv_questions = json.load(f)
with open("test/github_readme.json", "r", encoding="utf-8") as f:
    github_questions = json.load(f)

all_questions = arxiv_questions + github_questions
print(f"✅ Loaded {len(data)} chunks and {len(all_questions)} questions.")

# ==================== 2. 构建 BM25 (仅针对 README) ====================
print("🔹 Building BM25 index for READMEs...")
readme_corpus = [data[i]["text"] for i in readme_indices]
tokenized_corpus = [doc.lower().split() for doc in readme_corpus]
bm25 = BM25Okapi(tokenized_corpus)
# BM25 的索引 i 对应 readme_indices[i]
print(f"✅ BM25 index built for {len(readme_corpus)} README chunks.")

# ==================== 3. Router 逻辑 ====================
def is_code_query(query: str) -> bool:
    """简单路由：判断是否查代码/Repo"""
    keywords = [
        "code", "python", "github", "repo", "implementation", 
        "install", "usage", "api", "function", "class", "library",
        "pip", "clone", "setup"
    ]
    return any(k in query.lower() for k in keywords)

latency_stats = {"baseline": [], "advanced": []}

# ==================== 4. 检索函数 (修复版: Soft Routing) ====================

def retrieve_partitioned_baseline(query: str, top_k: int = 10) -> List[Dict]:
    """
    Baseline: 同时搜索两个索引，合并结果
    """
    t_start = time.time()
    query_vec = bi_encoder.encode([query], normalize_embeddings=True)
    
    # 1. 搜索 Paper Index
    D_p, I_p = index_paper.search(query_vec.astype(np.float32), top_k)
    
    # 2. 搜索 README Index
    D_r, I_r = index_readme.search(query_vec.astype(np.float32), top_k)
    
    all_results = []
    
    # 处理 Paper 结果
    for rank, (local_idx, score) in enumerate(zip(I_p[0], D_p[0])):
        if local_idx < 0: continue
        global_idx = paper_local_to_global[local_idx]
        all_results.append({
            "chunk_idx": global_idx,
            "score": float(score),
            "source": "paper"
        })

    # 处理 README 结果
    for rank, (local_idx, score) in enumerate(zip(I_r[0], D_r[0])):
        if local_idx < 0: continue
        global_idx = readme_local_to_global[local_idx]
        all_results.append({
            "chunk_idx": global_idx,
            "score": float(score),
            "source": "readme"
        })
    
    # 3. 合并并排序 (取 Top-K)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_results[:top_k]
    
    # 4. 格式化
    formatted_results = []
    for rank, item in enumerate(top_results):
        chunk = data[item["chunk_idx"]]
        formatted_results.append({
            "paper_id": chunk["metadata"].get("filename", "unknown"),
            "chunk_id": f"chunk_{item['chunk_idx']}",
            "score": item["score"],
            "text": chunk["text"],
            "rank": rank + 1
        })
        
    t_end = time.time()
    latency_stats["baseline"].append((t_end - t_start) * 1000)
    return formatted_results


def retrieve_partitioned_advanced(query: str, top_k: int = 10) -> List[Dict]:
    """
    Advanced: 
    1. 搜索两个索引 (Top-50 each)
    2. 合并候选集
    3. Rerank (Cross-Encoder 会自动挑选最好的)
    """
    t_start = time.time()
    candidate_ids: Set[int] = set()
    target_is_code = is_code_query(query)
    
    query_vec = bi_encoder.encode([query], normalize_embeddings=True)
    dense_k = 50 
    
    # 1. 搜 Paper (总是搜!)
    D_p, I_p = index_paper.search(query_vec.astype(np.float32), dense_k)
    for local_idx in I_p[0]:
        if local_idx >= 0: candidate_ids.add(paper_local_to_global[local_idx])

    # 2. 搜 README (总是搜!)
    D_r, I_r = index_readme.search(query_vec.astype(np.float32), dense_k)
    for local_idx in I_r[0]:
        if local_idx >= 0: candidate_ids.add(readme_local_to_global[local_idx])
            
    # 3. BM25 (仅针对 README，辅助召回)
    # 只有当是 Code Query 时才加 BM25，或者总是加
    if target_is_code: 
        bm25_k = 50
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25 = np.argsort(bm25_scores)[::-1][:bm25_k]
        for bm25_idx in top_bm25:
            if bm25_scores[bm25_idx] > 0:
                global_idx = readme_indices[bm25_idx]
                candidate_ids.add(global_idx)

    candidates = list(candidate_ids)
    if not candidates: return []

    # --- Stage 2: Reranking ---
    # Cross-Encoder 是最聪明的 Router，它会自己看哪个结果好
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

# ==================== 5. 准备评估数据 (保持不变) ====================
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
retrieve_partitioned_baseline("warm up query", top_k=10)
retrieve_partitioned_advanced("warm up query", top_k=10)
latency_stats["baseline"] = []
latency_stats["advanced"] = []

print("🔍 Running Partitioned Baseline (Router + Dense)...")
runs_baseline = {}
for qid, q_data in queries.items():
    runs_baseline[qid] = retrieve_partitioned_baseline(q_data["query"], top_k=10)

print("🔍 Running Partitioned Advanced (Router + Hybrid + Rerank)...")
runs_advanced = {}
for qid, q_data in queries.items():
    runs_advanced[qid] = retrieve_partitioned_advanced(q_data["query"], top_k=10)

print(f"✅ Retrieved results for {len(queries)} queries\n")

# ==================== 7. 输出延迟对比 ====================
print("=" * 80)
print("⚡ LATENCY COMPARISON (ms)")
print("=" * 80)

def print_latency(name, values):
    if not values: return
    avg = sum(values) / len(values)
    p95 = np.percentile(values, 95)
    print(f"{name:25s} | Avg: {avg:6.2f}ms | P95: {p95:6.2f}ms")

print_latency("Partitioned Baseline", latency_stats["baseline"])
print_latency("Partitioned Advanced", latency_stats["advanced"])
print("=" * 80 + "\n")

# ==================== 8. 评估指标对比 ====================
print("📊 Evaluating Baseline...")
results_baseline = evaluate(
    queries=queries,
    runs=runs_baseline,
    gold_paper=gold_paper,
    gold_passage=gold_passage,
    k_list=[5, 10],
    bootstrap=False
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
print("📈 PERFORMANCE COMPARISON (Partitioned Indices)")
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
with open("evaluation_partitioned.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"💾 Detailed results saved to: evaluation_partitioned.json")
