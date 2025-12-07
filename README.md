# CSE 291A Project - ArXiv Paper Search

## Environment Setup

This project uses a conda virtual environment to manage dependencies.

### Option 1: Using the existing environment

If the `cse291a` environment is already created:

```bash
conda activate cse291a
```

### Option 2: Create from scratch

```bash
# Create environment
conda create -n cse291a python=3.10 -y

# Activate environment
conda activate cse291a

# Install all dependencies
conda install -c conda-forge pandas numpy datasets sentence-transformers faiss-cpu pytorch -y
```

### Option 3: Create from environment.yml

```bash
conda env create -f environment.yml
conda activate cse291a
```

## Running the Scripts

**Important**: Always activate the environment first!

```bash
conda activate cse291a
```

### Step 1: Ingest ArXiv Data
```bash
python scripts/step1_ingest.py
```

### Step 2: Create Schema and Chunks
```bash
python scripts/step2_schema.py
```

### Step 3: Generate BGE Embeddings
```bash
python scripts/step3_embed_bge.py
```

### Step 4: Build RAG Index
```bash
python build_embedding.py
```
This now builds two separate indexes (papers vs GitHub READMEs) and prepends metadata (title/authors for papers, title for READMEs) before embedding:
- `rag_paper_index.faiss` / `rag_paper_embeddings.npy`
- `rag_readme_index.faiss` / `rag_readme_embeddings.npy`

### Step 5: Create Test Requests
```bash
python create_request.py
```

### Step 6: Evaluate Retrieval
```bash
python evaluate_retrieval.py
```

### Step 7: Prepare data for RAG pipeline
```bash
python convert_pdf_to_chunk_data.py
```

### Step 8: Start vLLM Server (Required for RAG Pipeline)
Before running the RAG pipeline, you need to start the vLLM server in a separate terminal:

```bash
# Make sure you're in the cse291a environment
conda activate cse291a

# Start the vLLM server (this will take a while to load the model)
vllm serve Qwen/Qwen3-0.6B --port 8082
```


### Step 9: Run the RAG Pipeline
In a new terminal (with the vLLM server still running):
```bash
conda activate cse291a
cd CSE_291A
python rag_pipeline.py
```

## Project Structure

```
project/
├── data/
│   ├── raw/
│   │   └── arxiv_meta_snapshot.jsonl
│   ├── logs/
│   │   └── ingest_report.json
│   ├── paper_meta.parquet
│   ├── paper_chunks.parquet
│   └── paper_links.parquet
├── index/
│   ├── arxiv_abs_bge_ip.faiss
│   └── lookup.parquet
├── scripts/
│   ├── step1_ingest.py
│   ├── step2_schema.py
│   ├── step3_embed_bge.py
│   └── search_bge.py
├── build_embedding.py
├── create_request.py
├── evaluate_retrieval.py
├── rag_evaluator.py
├── rag_pipeline.py
├── environment.yml
└── README.md
```

## Package Versions

- Python: 3.10
- PyTorch: 2.4.1
- sentence-transformers: 5.1.2
- transformers: 4.48.3
- datasets: 3.6.0
- faiss-cpu: 1.9.0
- pandas: 2.3.3
- numpy: 1.26.4

## Troubleshooting

### If you get import errors:
1. Make sure the environment is activated: `conda activate cse291a`
2. Verify installation: `conda list`

### CUDA/GPU Compatibility Issues (RTX 5070 and newer GPUs):
If you encounter CUDA errors like "no kernel image is available for execution on the device", 
see [INSTALL_CUDA_PYTORCH.md](INSTALL_CUDA_PYTORCH.md) for detailed instructions on updating PyTorch.

### To remove and recreate the environment:
```bash
conda deactivate
conda env remove -n cse291a
conda env create -f environment.yml
```

### To export your current environment:
```bash
conda env export > environment.yml
```
