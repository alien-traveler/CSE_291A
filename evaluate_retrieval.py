# -*- coding: utf-8 -*-
"""
RAG Retrieval Evaluation Script - Refactored
==========================================
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rag_evaluator import evaluate
from typing import Dict, List


# ==================== 加载数据 ====================
print("🔹 Loading model and data...")
model = SentenceTransformer("BAAI/bge-m3", cache_folder='./model')
index = faiss.read_index("rag_index.faiss")

with open("rag_all.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open("arxiv_paper.json", "r", encoding="utf-8") as f:
    arxiv_questions = json.load(f)

with open("github_readme.json", "r", encoding="utf-8") as f:
    github_questions = json.load(f)

all_questions = arxiv_questions + github_questions
print(f"✅ Loaded {len(all_questions)} test questions\n")


# ==================== 准备评估数据 ====================
def prepare_evaluation_data(questions: List[Dict]) -> tuple:
    """
    构建评估数据结构
    
    ✅ 改进:
    1. 根据匹配的reference_content数量进行分级打分
    2. 去除停用词,提高关键词匹配准确性
    3. 支持多reference_content的覆盖率评估
    """
    
    # 定义停用词
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'of', 'in', 'to', 'for', 'and', 'or', 'but', 'with', 'by', 'at',
        'from', 'as', 'on', 'this', 'that', 'these', 'those', 'it', 'its'
    }
    
    queries = {}
    gold_paper = {}
    gold_passage = {}
    
    for idx, q in enumerate(questions):
        if "question" not in q:
            continue
            
        qid = f"q{idx}"
        question = q["question"]
        ref_contents = q.get("reference_content", [])
        source_file = q["file"]
        
        # 添加 key_terms 用于 HasAnswer 评估
        queries[qid] = {
            "query": question,
            "key_terms": ref_contents  # 用于评估,不用于检索
        }
        
        gold_paper[qid] = {}
        gold_passage[qid] = {}
        
        # 标记相关文档和段落
        for chunk_idx, chunk in enumerate(data):
            meta = chunk["metadata"]
            chunk_file = meta.get("filename", "")
            chunk_id = f"chunk_{chunk_idx}"
            paper_id = chunk_file
            
            # 文件名匹配 = 相关
            if source_file in chunk_file or chunk_file in source_file:
                if paper_id not in gold_paper[qid]:
                    gold_paper[qid][paper_id] = 3
                
                # ✅ 改进的内容匹配逻辑
                if ref_contents:
                    chunk_text = chunk["text"].lower()
                    
                    # 方法1: 精确短语匹配
                    exact_matches = sum(
                        1 for ref in ref_contents 
                        if ref.lower() in chunk_text
                    )
                    
                    if exact_matches > 0:
                        # 有精确匹配,根据匹配数量打分
                        total_refs = len(ref_contents)
                        coverage = exact_matches / total_refs
                        
                        if coverage >= 0.8:
                            gold_passage[qid][chunk_id] = 3  # ⭐⭐⭐ 高度相关
                        elif coverage >= 0.5:
                            gold_passage[qid][chunk_id] = 3  # ⭐⭐⭐ 中高度相关
                        else:
                            gold_passage[qid][chunk_id] = 2  # ⭐⭐ 中度相关
                        continue
                    
                    # 方法2: 关键词匹配 (去除停用词)
                    keyword_matches = 0
                    
                    for ref in ref_contents:
                        # 提取实体词 (长度>3, 非停用词)
                        keywords = [
                            w for w in ref.lower().split() 
                            if len(w) > 3 and w not in STOPWORDS
                        ]
                        
                        if not keywords:
                            continue
                        
                        # 计算关键词覆盖率
                        chunk_tokens = set(chunk_text.split())
                        matched_keywords = sum(
                            1 for kw in keywords 
                            if kw in chunk_tokens
                        )
                        
                        keyword_coverage = matched_keywords / len(keywords)
                        
                        # 如果40%以上关键词匹配,认为该reference被覆盖
                        if keyword_coverage >= 0.4:
                            keyword_matches += 1
                    
                    # 根据匹配的reference数量打分
                    if keyword_matches > 0:
                        total_refs = len(ref_contents)
                        coverage = keyword_matches / total_refs
                        
                        if coverage >= 0.6:
                            gold_passage[qid][chunk_id] = 3  # ⭐⭐⭐ 高度相关
                        elif coverage >= 0.3:
                            gold_passage[qid][chunk_id] = 2  # ⭐⭐ 中度相关
                        else:
                            gold_passage[qid][chunk_id] = 1  # ⭐ 低度相关
                    else:
                        gold_passage[qid][chunk_id] = 1  # ⭐ 来自正确文档但内容不匹配
                else:
                    gold_passage[qid][chunk_id] = 2  # ⭐⭐ 无reference_content
    
    return queries, gold_paper, gold_passage


# ==================== 检索函数 ====================
def retrieve_for_query(query: str, top_k: int = 10) -> List[Dict]:
    """执行FAISS检索"""
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec.astype(np.float32), top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < 0 or idx >= len(data):
            continue
            
        chunk = data[idx]
        meta = chunk["metadata"]
        
        results.append({
            "paper_id": meta.get("filename", "unknown"),
            "chunk_id": f"chunk_{idx}",
            "score": float(score),
            "text": chunk["text"],
            "rank": rank + 1
        })
    
    return results


# ==================== 运行评估 ====================
print("🔍 Preparing evaluation data...")
queries, gold_paper, gold_passage = prepare_evaluation_data(all_questions)

print(f"✅ Prepared {len(queries)} queries")
print(f"   - Paper labels: {sum(len(v) for v in gold_paper.values())}")
print(f"   - Passage labels: {sum(len(v) for v in gold_passage.values())}")

# ✅ 新增: 显示相关性分布
passage_relevance_dist = {}
for qid in gold_passage:
    for chunk_id, score in gold_passage[qid].items():
        passage_relevance_dist[score] = passage_relevance_dist.get(score, 0) + 1

print(f"   - Passage relevance distribution:")
for score in sorted(passage_relevance_dist.keys(), reverse=True):
    print(f"     Score {score}: {passage_relevance_dist[score]} chunks")
print()

if len(queries) == 0:
    print("❌ No valid queries found!")
    exit(1)

print("🔍 Running retrieval...")
runs = {}
for qid, q_data in queries.items():
    runs[qid] = retrieve_for_query(q_data["query"], top_k=10)

print(f"✅ Retrieved results for {len(runs)} queries\n")

print("📊 Evaluating...")
results = evaluate(
    queries=queries,
    runs=runs,
    gold_paper=gold_paper,
    gold_passage=gold_passage,
    k_list=[1, 3, 5, 10],
    bootstrap=True,
    bootstrap_iters=1000
)

# ==================== 输出结果 ====================
print("\n" + "=" * 80)
print("📈 EVALUATION RESULTS")
print("=" * 80)

print("\n📊 Macro-Averaged Metrics:\n")

# ✅ 精简的指标分组
metric_groups = {
    "📄 Document Retrieval": [
        "paper_mrr@10",
        "paper_ndcg@5", 
        "paper_r@10"
    ],
    "📑 Passage Retrieval": [
        "passage_ndcg@5",
        "passage_p@5"
    ],
    "✅ Answer Quality": [
        "has_answer@5",
        "answer_coverage@5"
    ],
    "🎯 Overall Quality": [
        "cite_then_quote@5",
        "diversity@5"
    ]
}

for group_name, metric_list in metric_groups.items():
    print(f"\n{group_name}:")
    
    for metric_name in metric_list:
        if metric_name in results["macro"]:
            values = results["macro"][metric_name]
            mean = values["mean"]
            if "ci95" in values:
                ci_lo, ci_hi = values["ci95"]
                print(f"  {metric_name:25s}: {mean:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
            else:
                print(f"  {metric_name:25s}: {mean:.4f}")

print("\n" + "-" * 80)
print("\n📋 Sample Query Analysis (q0):\n")

# 显示第一个查询的详细结果
if "q0" in results["per_query"]:
    q0_result = results["per_query"]["q0"]
    q0_query = queries["q0"]
    
    print(f"Query: {q0_query['query']}")
    print(f"Reference contents: {len(q0_query['key_terms'])} items\n")
    
    # ✅ 只显示 @5 的核心指标
    print("Core Metrics @ k=5:")
    
    # Document Retrieval
    print(f"  📄 Paper MRR         : {q0_result['metrics'].get('paper_mrr@5', 0):.2f}")
    print(f"  📄 Paper nDCG        : {q0_result['metrics'].get('paper_ndcg@5', 0):.2f}")
    
    # Passage Retrieval
    print(f"  📑 Passage nDCG      : {q0_result['metrics'].get('passage_ndcg@5', 0):.2f}")
    print(f"  📑 Passage Precision : {q0_result['metrics'].get('passage_p@5', 0):.2f}")
    
    # Answer Quality
    if "has_answer@5" in q0_result["metrics"]:
        has_ans = q0_result["metrics"]["has_answer@5"]
        coverage = q0_result["metrics"]["answer_coverage@5"]
        found = int(coverage * len(q0_query['key_terms']))
        total = len(q0_query['key_terms'])
        
        print(f"  ✅ HasAnswer         : {has_ans:.2f}")
        print(f"  ✅ Answer Coverage   : {coverage:.2f}  ({found}/{total} references found)")
    
    # Overall Quality
    print(f"  🎯 CiteThenQuote     : {q0_result['metrics'].get('cite_then_quote@5', 0):.2f}")
    print(f"  🎯 Diversity         : {q0_result['metrics'].get('diversity@5', 0):.2f}")

# ==================== 保存结果 ====================
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to: evaluation_results.json")
print("=" * 80 + "\n")