# -*- coding: utf-8 -*-
"""
Crawl README.md files from GitHub repositories
----------------------------------------------
Example:
  python crawl_github_readme.py --query "vision language model" --language python --limit 100
"""
import os
import requests
import json
from tqdm import tqdm
from urllib.parse import quote
import argparse

def crawl_github_readmes(query="vision language model", language="python", limit=50, save_dir="github_readmes", token=None):
    os.makedirs(save_dir, exist_ok=True)
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # Step 1: 搜索仓库
    search_url = f"https://api.github.com/search/repositories?q={quote(query)}+language:{language}&sort=stars&order=desc&per_page={limit}"
    print(f"🔍 Searching GitHub for: {query} ({language})")
    repos = requests.get(search_url, headers=headers).json().get("items", [])
    print(f"✅ Found {len(repos)} repositories")

    # Step 2: 下载 README.md
    results = []
    for repo in tqdm(repos, desc="Downloading READMEs"):
        full_name = repo["full_name"]
        default_branch = repo["default_branch"]
        stars = repo["stargazers_count"]
        html_url = repo["html_url"]

        readme_url = f"https://raw.githubusercontent.com/{full_name}/{default_branch}/README.md"
        resp = requests.get(readme_url, headers=headers)

        if resp.status_code == 200 and len(resp.text) > 100:
            filename = full_name.replace("/", "_") + "_README.md"
            save_path = os.path.join(save_dir, filename)
            with open(save_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(resp.text)
            results.append({
                "repo": full_name,
                "stars": stars,
                "url": html_url,
                "readme_path": save_path
            })
        else:
            print(f"⚠️ No valid README found for {full_name}")

    # Step 3: 保存元数据
    meta_path = os.path.join(save_dir, "readme_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(results)} READMEs and metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="multimodal transformer")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--token", type=str, default=os.getenv("GITHUB_TOKEN"))
    args = parser.parse_args()

    crawl_github_readmes(args.query, args.language, args.limit, token=args.token)
