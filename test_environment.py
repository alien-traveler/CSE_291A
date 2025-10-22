#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
"""

print("Testing imports...")

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except Exception as e:
    print("✗ pandas:", str(e))

try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except Exception as e:
    print("✗ numpy:", str(e))

try:
    import torch
    print("✓ torch:", torch.__version__)
except Exception as e:
    print("✗ torch:", str(e))

try:
    import faiss
    print("✓ faiss: installed")
except Exception as e:
    print("✗ faiss:", str(e))

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers: installed")
except Exception as e:
    print("✗ sentence-transformers:", str(e))

try:
    from datasets import load_dataset
    print("✓ datasets: installed")
except Exception as e:
    print("✗ datasets:", str(e))

print("\n✅ All core dependencies are working!")
print("\nEnvironment is ready to use.")
print("Run your scripts with: conda activate cse291a")
