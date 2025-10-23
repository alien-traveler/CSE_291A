import feedparser
import time
import json
from tqdm import tqdm
import urllib.parse


def fetch_arxiv_keyword(
        keywords=["large language model", "visual language model", "vision language model"],
        categories=["cs.AI", "cs.CL", "cs.CV"],
        max_results=1000,
        save_path="arxiv_llm_vlm.jsonl"
):
    base_url = "http://export.arxiv.org/api/query?"
    results_per_call = 200
    papers = []

    # 构造关键词部分
    keyword_query = " OR ".join([f'ti:"{k}" OR abs:"{k}"' for k in keywords])
    category_query = " OR ".join([f"cat:{c}" for c in categories])
    full_query = f"({keyword_query}) AND ({category_query})"
    encoded_query = urllib.parse.quote(full_query)

    print("🔍 Query:", full_query)

    for start in tqdm(range(0, max_results, results_per_call)):
        url = f"{base_url}search_query={encoded_query}&start={start}&max_results={results_per_call}&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)

        if len(feed.entries) == 0:
            break

        for entry in feed.entries:
            paper = {
                "id": entry.id.split("/")[-1],
                "title": entry.title.replace("\n", " ").strip(),
                "summary": entry.summary.replace("\n", " ").strip(),
                "authors": [a.name for a in entry.authors],
                "published": entry.published,
                "updated": entry.updated,
                "link_pdf": next((l.href for l in entry.links if l.rel == "related"), None),
                "link_page": entry.link,
            }
            papers.append(paper)

        time.sleep(1)

    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        for p in papers:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved {len(papers)} LLM/VLM-related papers to {save_path}")


# 示例调用
fetch_arxiv_keyword(
    keywords=["large language model", "visual language model", "vision language model", "multimodal large model"],
    categories=["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE"],
    max_results=2000,
    save_path="arxiv_llm_vlm_2025.jsonl"
)
