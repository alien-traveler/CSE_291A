import os
import json
import time
import requests
from tqdm import tqdm

def download_arxiv_pdfs(jsonl_path="arxiv_llm_vlm_2025.jsonl", save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        papers = [json.loads(line) for line in f if line.strip()]

    for p in tqdm(papers, desc="Downloading PDFs"):
        paper_id = p["id"]
        pdf_url = p["link_pdf"]

        # 如果没有 related 链接，就构造默认 PDF 链接
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        save_path = os.path.join(save_dir, f"{paper_id}.pdf")
        if os.path.exists(save_path):
            continue  # 跳过已下载文件

        try:
            resp = requests.get(pdf_url, timeout=20)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                with open(save_path, "wb") as f:
                    f.write(resp.content)
            else:
                print(f"⚠️ Failed to download {paper_id} ({resp.status_code})")
        except Exception as e:
            print(f"❌ Error downloading {paper_id}: {e}")
        time.sleep(1)  # 避免请求过快

    print(f"✅ Done! PDFs saved in {os.path.abspath(save_dir)}")


# 示例调用
if __name__ == "__main__":
    download_arxiv_pdfs("arxiv_llm_vlm_2025.jsonl", save_dir="data")
