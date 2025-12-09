# -*- coding: utf-8 -*-
"""
RAG Retrieval Inspector - 查看第一个查询的详细检索结果
=======================================================
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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


# ==================== 检索函数 ====================
def retrieve_for_query(query: str, top_k: int = 10) -> List[Dict]:
    """执行FAISS检索并返回详细结果"""
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec.astype(np.float32), top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        # ✅ 修复: 转换为Python原生类型
        idx = int(idx)  # numpy.int64 → int
        score = float(score)  # numpy.float32 → float
        
        if idx < 0 or idx >= len(data):
            continue
            
        chunk = data[idx]
        meta = chunk["metadata"]
        
        results.append({
            "rank": rank + 1,  # 已经是int,不需要转换
            "chunk_idx": idx,
            "paper_id": meta.get("filename", "unknown"),
            "score": score,
            "text": chunk["text"],
            "page": meta.get("page", "N/A"),
            "chunk_id": meta.get("chunk_id", "N/A")
        })
    
    return results


# ==================== 获取第一个查询 ====================
if len(all_questions) == 0:
    print("❌ No questions found!")
    exit(1)

first_question = all_questions[0]

print("=" * 80)
print("📝 FIRST QUERY (q0)")
print("=" * 80)
print(f"\n🔍 Query: {first_question['question']}")
print(f"\n📄 Source File: {first_question['file']}")

if "reference_content" in first_question and first_question["reference_content"]:
    print(f"\n✅ Reference Content ({len(first_question['reference_content'])} items):")
    for i, ref in enumerate(first_question['reference_content'], 1):
        print(f"   {i}. {ref}")
else:
    print("\n⚠️  No reference content available")

print("\n" + "=" * 80)


# ==================== 执行检索 ====================
print("\n🔍 Running retrieval (Top-10)...\n")
results = retrieve_for_query(first_question['question'], top_k=10)


# ==================== 显示检索到的论文 ====================
print("=" * 80)
print("📄 RETRIEVED PAPERS (Unique)")
print("=" * 80)

# 统计每个论文的chunk数量和最高分数
paper_stats = {}
for r in results:
    paper_id = r["paper_id"]
    if paper_id not in paper_stats:
        paper_stats[paper_id] = {
            "count": 0,
            "max_score": 0.0,
            "ranks": []
        }
    paper_stats[paper_id]["count"] += 1
    paper_stats[paper_id]["max_score"] = max(paper_stats[paper_id]["max_score"], r["score"])
    paper_stats[paper_id]["ranks"].append(r["rank"])

# 按最高分数排序
sorted_papers = sorted(
    paper_stats.items(), 
    key=lambda x: x[1]["max_score"], 
    reverse=True
)

correct_paper = first_question['file']

for i, (paper_id, stats) in enumerate(sorted_papers, 1):
    is_correct = "✅" if correct_paper in paper_id or paper_id in correct_paper else "❌"
    print(f"\n{i}. {is_correct} {paper_id}")
    print(f"   - Chunks in Top-10: {stats['count']}")
    print(f"   - Max Score: {stats['max_score']:.4f}")
    print(f"   - Ranks: {', '.join(map(str, sorted(stats['ranks'])))}")


# ==================== 显示检索到的段落 ====================
print("\n" + "=" * 80)
print("📑 RETRIEVED CHUNKS (Top-10)")
print("=" * 80)

for r in results:
    is_correct_paper = "✅" if correct_paper in r["paper_id"] or r["paper_id"] in correct_paper else "❌"
    
    print(f"\n【Rank {r['rank']}】 {is_correct_paper} Score: {r['score']:.4f}")
    print(f"Paper: {r['paper_id']}")
    print(f"Chunk ID: chunk_{r['chunk_idx']} (Page: {r['page']})")
    print(f"Text Preview:")
    
    # 显示文本片段 (最多300字符)
    text = r["text"]
    if len(text) > 300:
        print(f"  {text[:300]}...")
    else:
        print(f"  {text}")
    
    # 检查是否包含reference content
    if "reference_content" in first_question and first_question["reference_content"]:
        matched_refs = []
        text_lower = text.lower()
        for i, ref in enumerate(first_question['reference_content'], 1):
            if ref.lower() in text_lower:
                matched_refs.append(i)
        
        if matched_refs:
            print(f"  💡 Contains references: {matched_refs}")


# ==================== 保存详细结果到文件 ====================
output = {
    "query": first_question['question'],
    "source_file": first_question['file'],
    "reference_content": first_question.get('reference_content', []),
    "papers": [
        {
            "paper_id": paper_id,
            "is_correct": correct_paper in paper_id or paper_id in correct_paper,
            "chunk_count": stats["count"],
            "max_score": stats["max_score"],  # 已经是float
            "ranks": sorted(stats["ranks"])  # 已经是int
        }
        for paper_id, stats in sorted_papers
    ],
    "chunks": results  # 所有值都已转换为Python原生类型
}

# ✅ 现在可以安全地序列化了
with open("q0_retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("💾 Detailed results saved to: q0_retrieval_results.json")
print("=" * 80 + "\n")


# ==================== 统计信息 ====================
print("=" * 80)
print("📊 SUMMARY STATISTICS")
print("=" * 80)

correct_chunks = sum(
    1 for r in results 
    if correct_paper in r["paper_id"] or r["paper_id"] in correct_paper
)

print(f"\n✅ Correct Paper Chunks in Top-10: {correct_chunks}/10")
print(f"📄 Total Unique Papers: {len(sorted_papers)}")
print(f"🎯 Correct Paper in Results: {'Yes ✅' if any(correct_paper in p or p in correct_paper for p in paper_stats.keys()) else 'No ❌'}")

if correct_chunks > 0:
    correct_ranks = [
        r["rank"] for r in results 
        if correct_paper in r["paper_id"] or r["paper_id"] in correct_paper
    ]
    print(f"📍 Correct Chunks at Ranks: {', '.join(map(str, sorted(correct_ranks)))}")
    print(f"🥇 Best Rank of Correct Chunk: {min(correct_ranks)}")

print("\n" + "=" * 80 + "\n")