"""Resource loaders for MemeMatcher application"""
import streamlit as st
import pickle
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from src.data_loader import load_memes
from components.config import (
    DATA_PATH,
    BM25_INDEX_PATH,
    SBERT_USAGE_INDEX_PATH,
    SBERT_VISUAL_INDEX_PATH,
    SBERT_MEME_IDS_PATH,
    SBERT_MODEL_NAME,
    CLIP_INDEX_PATH,
    CLIP_MEME_IDS_PATH,
    CLIP_MODEL_NAME
)


@st.cache_data
def load_data():
    """Load memes data and create lookup dictionary
    
    Returns:
        tuple: (memes list, memes_by_id dict)
    """
    memes = load_memes(DATA_PATH)
    memes_by_id = {m["meme_id"]: m for m in memes}
    return memes, memes_by_id


@st.cache_resource
def load_bm25_index():
    """Load pickled BM25 index
    
    Returns:
        dict: BM25 index data structure
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_faiss_indices():
    """Load FAISS indices for SBERT semantic search
    
    Returns:
        tuple: (index_usage, index_visual, meme_ids)
    """
    idx_usage = faiss.read_index(SBERT_USAGE_INDEX_PATH)
    idx_visual = faiss.read_index(SBERT_VISUAL_INDEX_PATH)
    with open(SBERT_MEME_IDS_PATH, "r", encoding="utf-8") as f:
        meme_ids = json.load(f)
    return idx_usage, idx_visual, meme_ids


@st.cache_resource
def load_clip_index():
    """Load FAISS index for CLIP image search
    
    Returns:
        tuple: (clip_index, clip_meme_ids)
    """
    idx = faiss.read_index(CLIP_INDEX_PATH)
    with open(CLIP_MEME_IDS_PATH, "r", encoding="utf-8") as f:
        meme_ids = json.load(f)
    return idx, meme_ids


@st.cache_resource
def get_sbert_model():
    """Load and cache SBERT model (lazy - only when called)
    
    Returns:
        SentenceTransformer: SBERT model for semantic search
    """
    with st.spinner("Loading SBERT model..."):
        return SentenceTransformer(SBERT_MODEL_NAME)


@st.cache_resource
def get_clip_model_and_processor():
    """Load and cache CLIP model and processor (lazy - only when called)
    
    Returns:
        tuple: (model, processor, device)
    """
    with st.spinner("Loading CLIP model..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        # Use fast image processor if torchvision is available; otherwise fall back safely
        try:
            import torchvision  # noqa: F401
            use_fast = True
        except Exception:
            use_fast = False
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=use_fast)
        model.eval()
        return model, processor, device
