import os
import json
from tqdm import tqdm
import fitz  # pip install pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 提取纯文本
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()


def clean_text(text: str) -> str:
    """
    简单清洗函数：
    - 去掉空行、页眉页脚
    - 截断 References 后的部分
    """
    lines = text.splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    # 找到 References 并截断
    for i, line in enumerate(lines):
        if line.lower().startswith("references"):
            lines = lines[:i]
            break

    return " ".join(lines)


def split_text(text: str, chunk_size: int = 1024, overlap: int = 200):
    """
    使用 LangChain Text Splitter 进行语义切块
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def pdfs_to_jsonl(pdf_dir="data", save_path="rag_data.jsonl"):
    """
    主函数：批量提取 PDF → 文本 → 分块 → 保存为 JSONL
    """
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    all_chunks = []
    for f in tqdm(pdf_files, desc="Processing PDFs"):
        paper_id = os.path.splitext(f)[0]
        pdf_path = os.path.join(pdf_dir, f)
        try:
            text = extract_text_from_pdf(pdf_path)
            text = clean_text(text)
            chunks = split_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{paper_id}_{i}",
                    "paper_id": paper_id,
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "chunk_id": i,
                        "filename": f
                    }
                })

        except Exception as e:
            print(f"⚠️ Failed to process {f}: {e}")

    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        for item in all_chunks:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Done! {len(all_chunks)} chunks saved to {save_path}")


if __name__ == "__main__":
    pdfs_to_jsonl(pdf_dir="data", save_path="rag_data.jsonl")

