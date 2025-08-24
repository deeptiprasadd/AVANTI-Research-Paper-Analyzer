# test.py
try:
    import streamlit as st
    print("✅ Streamlit OK")
    
    from transformers import pipeline
    print("✅ Transformers OK")
    
    import torch
    print("✅ PyTorch OK")
    
    print("🎉 All packages working!")
    
except ImportError as e:
    print(f"❌ Missing: {e}")

