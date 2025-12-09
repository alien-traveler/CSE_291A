#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Request Generator for RAG Testing
========================================
This script generates test queries for RAG evaluation by:
1. Randomly selecting 10 papers or repositories
2. Using OpenAI GPT-4o to analyze content and generate relevant questions
3. Saving results in JSON format with questions, reference content, and file paths
"""

import os
import json
import random
import fitz  # PyMuPDF for PDF processing
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAIKEY'))

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file
    """
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """
    Clean text by removing empty lines and truncating after References
    """
    lines = text.splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    
    # Find References and truncate
    for i, line in enumerate(lines):
        if line.lower().startswith("references"):
            lines = lines[:i]
            break
    
    return " ".join(lines)

def read_readme_file(readme_path: str) -> str:
    """
    Read content from README markdown file
    """
    try:
        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {readme_path}: {e}")
        return ""

def generate_question_with_gpt4o(content: str, file_type: str, file_path: str) -> Dict[str, Any]:
    """
    Use GPT-4o to generate a question and reference content based on the input content
    """
    try:
        # For PDF: truncate at References section (consistent with convert_pdf_to_chunk_data.py)
        # For README: use full content (consistent with convert_readme_to_chunk_data.py)
        if file_type == "pdf":
            # PDF content is already cleaned (References truncated) in clean_text function
            # Just ensure it's not too long for API
            max_length = 8000  # Increased limit for better context
            if len(content) > max_length:
                content = content[:max_length] + "..."
        # For README, use full content as it's stored as single unit in the database
        
        prompt = f"""
You are an expert in analyzing technical content for RAG (Retrieval-Augmented Generation) evaluation. 

Given the following {file_type} content, please:
1. Generate ONE specific, well-formed question that can be answered using this content
2. Identify 1-3 relevant text segments from the content that directly relate to the question
3. Each text segment should be a continuous piece of text that helps answer the question

The question should be SPECIFIC and test retrieval capabilities, not general questions. Examples:

For research papers:
- Instead of "What is the main contribution?" ask "What specific matrices does LoRA introduce for parameter-efficient fine-tuning?"
- Instead of "What is the methodology?" ask "What is the rank parameter r used for in LoRA decomposition?"
- Instead of "What are the results?" ask "What is the accuracy improvement achieved on the GLUE benchmark?"

For code repositories (README files):
- Instead of "What is the main purpose?" ask "What specific loss function is used in the training process?"
- Instead of "What are the key features?" ask "What are the required dependencies listed in requirements.txt?"
- Instead of "How to use?" ask "What is the exact command to run the training script?"
- Instead of "What files are required?" ask "What is the specific evaluation code for [Project Name] and how should it be run?"
- Instead of "Where to download?" ask "What are the specific installation steps for [Project Name] and what dependencies are needed?"

IMPORTANT: For README files, use the project name/title from the content in your question. For example:
- "What is the specific evaluation code for MADTP and how should it be run?"
- "What are the installation requirements for BitNet and what dependencies are needed?"
- "What is the training command for Transflower and what parameters are required?"

Content:
{content}

Please respond with a JSON object in this exact format:
{{
    "question": "Your specific question here",
    "reference_content": [
        "First relevant text segment",
        "Second relevant text segment (if applicable)",
        "Third relevant text segment (if applicable)"
    ]
}}

Make sure the question is:
- Specific and answerable from the content
- Tests retrieval of specific technical details
- Clear and well-formed
- Focused on concrete information that would be found in the text
- For README files, includes the project name/title in the question

The reference_content should contain the most relevant parts of the text that directly support answering the question.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in technical content analysis and question generation for RAG evaluation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the JSON response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        elif "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_text = response_text[json_start:json_end]
        else:
            raise ValueError("No valid JSON found in response")
        
        result = json.loads(json_text)
        
        # Add file path to the result
        result["file"] = file_path
        
        return result
        
    except Exception as e:
        print(f"Error generating question with GPT-4o: {e}")
        # Return a fallback structure
        return {
            "question": f"What is the main topic discussed in this {file_type}?",
            "reference_content": [content[:500] + "..." if len(content) > 500 else content],
            "file": file_path
        }

def get_random_files(num_files: int = 20) -> List[Dict[str, str]]:
    """
    Get random files from both data and github_readmes directories
    """
    files = []
    
    # Get PDF files from data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files[:10]:  # Take first 10 PDFs
            files.append({
                "path": os.path.join(data_dir, pdf_file),
                "type": "pdf",
                "name": pdf_file
            })
    
    # Get README files from github_readmes directory
    readme_dir = "github_readmes"
    if os.path.exists(readme_dir):
        readme_files = [f for f in os.listdir(readme_dir) if f.endswith('.md') and f != 'readme_metadata.json']
        for readme_file in readme_files[:10]:  # Take first 10 READMEs
            files.append({
                "path": os.path.join(readme_dir, readme_file),
                "type": "readme",
                "name": readme_file
            })
    
    # Randomly select files
    selected_files = random.sample(files, min(num_files, len(files)))
    return selected_files

def main():
    """
    Main function to generate test queries
    """
    print("🚀 Starting RAG Test Query Generation...")
    
    # Get random files
    selected_files = get_random_files(20)
    print(f"📁 Selected {len(selected_files)} files for processing")
    
    pdf_results = []
    readme_results = []
    
    for i, file_info in enumerate(tqdm(selected_files, desc="Processing files")):
        file_path = file_info["path"]
        file_type = file_info["type"]
        file_name = file_info["name"]
        
        print(f"\n📄 Processing {i+1}/20: {file_name}")
        
        # Extract content based on file type
        if file_type == "pdf":
            content = extract_text_from_pdf(file_path)
            content = clean_text(content)
        elif file_type == "readme":
            content = read_readme_file(file_path)
        else:
            print(f"⚠️ Unknown file type: {file_type}")
            continue
        
        if not content or len(content.strip()) < 100:
            print(f"⚠️ Skipping {file_name}: content too short or empty")
            continue
        
        # Generate question using GPT-4o
        print(f"🤖 Generating question for {file_name}...")
        result = generate_question_with_gpt4o(content, file_type, file_path)
        
        # Separate results by type
        if file_type == "pdf":
            pdf_results.append(result)
        else:
            readme_results.append(result)
        
        print(f"✅ Generated question: {result['question'][:100]}...")
    
    # Save results to separate files
    os.makedirs("test", exist_ok=True)
    
    # Save PDF results
    if pdf_results:
        pdf_output_path = "test/arxiv_paper.json"
        with open(pdf_output_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_results, f, ensure_ascii=False, indent=2)
        print(f"📁 PDF results saved to: {pdf_output_path}")
    
    # Save README results
    if readme_results:
        readme_output_path = "test/github_readme.json"
        with open(readme_output_path, 'w', encoding='utf-8') as f:
            json.dump(readme_results, f, ensure_ascii=False, indent=2)
        print(f"📁 README results saved to: {readme_output_path}")
    
    print(f"\n🎉 Successfully generated {len(pdf_results)} PDF queries and {len(readme_results)} README queries!")
    
    # Print summary
    print("\n📊 PDF Papers Summary:")
    for i, result in enumerate(pdf_results, 1):
        print(f"{i}. {result['file']}")
        print(f"   Q: {result['question']}")
        print(f"   Ref segments: {len(result['reference_content'])}")
        print()
    
    print("\n📊 GitHub READMEs Summary:")
    for i, result in enumerate(readme_results, 1):
        print(f"{i}. {result['file']}")
        print(f"   Q: {result['question']}")
        print(f"   Ref segments: {len(result['reference_content'])}")
        print()

if __name__ == "__main__":
    main()
