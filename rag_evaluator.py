"""
RAG Retrieval Evaluator - Corrected Version
============================================
"""
from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Tuple


def _safe_mean(values: List[float]) -> float:
    """安全计算平均值"""
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values: List[float], iters: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap置信区间计算"""
    if not values:
        return (0.0, 0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(_safe_mean(sample))
    means.sort()
    lo_idx = max(0, int((alpha / 2) * iters) - 1)
    hi_idx = min(iters - 1, int((1 - alpha / 2) * iters) - 1)
    return (_safe_mean(values), means[lo_idx], means[hi_idx])


def _jaccard_similarity(text1: str, text2: str) -> float:
    """计算两个文本的Jaccard相似度"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2) or 1
    return intersection / union


def dcg(relevances: List[int]) -> float:
    """折损累积增益"""
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(pred_ids: List[str], gold_labels: Dict[str, int], k: int) -> float:
    """
    ✅ 修复: 归一化折损累积增益 @k
    
    理想排序应该是"检索到的这k个文档的最佳排序"
    而不是"所有gold_labels中最高的k个"
    """
    actual_rels = [gold_labels.get(pid, 0) for pid in pred_ids[:k]]
    ideal_rels = sorted(actual_rels, reverse=True)  # ✅ 修复点
    
    ideal_dcg = dcg(ideal_rels)
    actual_dcg = dcg(actual_rels)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(pred_ids: List[str], relevant_ids: Dict[str, int], k: int) -> float:
    """精确率 @k"""
    hits = sum(1 for pid in pred_ids[:k] if relevant_ids.get(pid, 0) >= 1)
    return hits / max(1, k)


def recall_at_k(pred_ids: List[str], relevant_ids: Dict[str, int], k: int) -> float:
    """召回率 @k"""
    total_relevant = sum(1 for v in relevant_ids.values() if v >= 1)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for pid in pred_ids[:k] if relevant_ids.get(pid, 0) >= 1)
    return hits / total_relevant


def mrr_at_k(pred_ids: List[str], relevant_ids: Dict[str, int], k: int) -> float:
    """平均倒数排名 @k"""
    for i, pid in enumerate(pred_ids[:k]):
        if relevant_ids.get(pid, 0) >= 1:
            return 1.0 / (i + 1)
    return 0.0


def has_answer_at_k(pred_texts: List[str], key_terms: List[str], k: int) -> float:
    """✅ 修复: 统一使用 40% 阈值"""
    if not key_terms:
        return 0.0
    combined_text = " ".join(pred_texts[:k]).lower()
    for term in key_terms:
        term_lower = term.lower()
        if term_lower in combined_text:
            return 1.0
        term_tokens = {w for w in term_lower.split() if len(w) > 3}
        if not term_tokens:
            continue
        text_tokens = set(combined_text.split())
        if len(term_tokens & text_tokens) / len(term_tokens) >= 0.4:  # ✅ 改为0.4
            return 1.0
    return 0.0


def answer_coverage_at_k(pred_texts: List[str], key_terms: List[str], k: int) -> float:
    """答案覆盖率"""
    if not key_terms:
        return 0.0
    combined_text = " ".join(pred_texts[:k]).lower()
    matched_count = 0
    for term in key_terms:
        term_lower = term.lower()
        if term_lower in combined_text:
            matched_count += 1
            continue
        term_tokens = {w for w in term_lower.split() if len(w) > 3}
        if not term_tokens:
            continue
        text_tokens = set(combined_text.split())
        overlap_ratio = len(term_tokens & text_tokens) / len(term_tokens)
        if overlap_ratio >= 0.4:
            matched_count += 1
    return matched_count / len(key_terms)


def diversity_at_k(pred_texts: List[str], k: int) -> float:
    """检索结果多样性"""
    texts = pred_texts[:k]
    if len(texts) < 2:
        return 1.0
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities.append(_jaccard_similarity(texts[i], texts[j]))
    return 1.0 - _safe_mean(similarities) if similarities else 1.0


def cite_then_quote_at_k(pred_items: List[Dict], relevant_papers: Dict[str, int], relevant_passages: Dict[str, int], k: int) -> float:
    """组合指标"""
    items = pred_items[:k]
    paper_hit = any(relevant_papers.get(item.get("paper_id", ""), 0) >= 1 for item in items)
    passage_hit = any(relevant_passages.get(item.get("chunk_id", ""), 0) >= 1 for item in items)
    return 1.0 if (paper_hit and passage_hit) else 0.0


def evaluate(queries: Dict[str, Dict[str, Any]], runs: Dict[str, List[Dict[str, Any]]], gold_paper: Dict[str, Dict[str, int]], gold_passage: Dict[str, Dict[str, int]], k_list: List[int] = [5, 10], bootstrap: bool = True, bootstrap_iters: int = 1000) -> Dict[str, Any]:
    """主评估函数"""
    per_query_results = {}
    
    for qid, retrieved_items in runs.items():
        query_info = queries.get(qid, {})
        key_terms = query_info.get("key_terms", [])
        paper_gold = gold_paper.get(qid, {})
        passage_gold = gold_passage.get(qid, {})
        
        paper_ids = [item.get("paper_id", "") for item in retrieved_items]
        passage_ids = [item.get("chunk_id", "") for item in retrieved_items]
        texts = [item.get("text", "") for item in retrieved_items]
        
        # ✅ 统一使用 >= 2 作为相关标准
        paper_relevant = {pid: 1 for pid, label in paper_gold.items() if label >= 2}
        passage_relevant = {cid: 1 for cid, label in passage_gold.items() if label >= 2}
        
        metrics = {}
        for k in k_list:
            metrics[f"paper_mrr@{k}"] = mrr_at_k(paper_ids, paper_relevant, k)
            metrics[f"paper_ndcg@{k}"] = ndcg_at_k(paper_ids, paper_gold, k)
            metrics[f"paper_r@{k}"] = recall_at_k(paper_ids, paper_relevant, k)
            if k == 5:
                metrics[f"passage_ndcg@{k}"] = ndcg_at_k(passage_ids, passage_gold, k)
                metrics[f"passage_p@{k}"] = precision_at_k(passage_ids, passage_relevant, k)
                metrics[f"has_answer@{k}"] = has_answer_at_k(texts, key_terms, k)
                metrics[f"answer_coverage@{k}"] = answer_coverage_at_k(texts, key_terms, k)
                metrics[f"cite_then_quote@{k}"] = cite_then_quote_at_k(retrieved_items, paper_relevant, passage_relevant, k)
                metrics[f"diversity@{k}"] = diversity_at_k(texts, k)
        per_query_results[qid] = {"metrics": metrics}
    
    macro_results = {}
    if per_query_results:
        sample_metrics = next(iter(per_query_results.values()))["metrics"]
        metric_names = list(sample_metrics.keys())
    else:
        metric_names = []
    
    for metric_name in metric_names:
        values = [per_query_results[qid]["metrics"][metric_name] for qid in per_query_results]
        if bootstrap and values:
            mean, ci_low, ci_high = _bootstrap_ci(values, iters=bootstrap_iters)
            macro_results[metric_name] = {"mean": mean, "ci95": [ci_low, ci_high]}
        else:
            macro_results[metric_name] = {"mean": _safe_mean(values)}
    
    return {"macro": macro_results, "per_query": per_query_results}