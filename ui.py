"""MemeMatcher - Meme Retrieval System with BM25, SBERT, and CLIP"""
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import components
from components.loaders import (
    load_data,
    load_bm25_index,
    load_faiss_indices,
    load_clip_index,
    get_sbert_model,
    get_clip_model_and_processor
)
from components.state import initialize_state, add_to_search_history
from components.styles import inject_styles
from components.ui_config_bar import render_config_bar
from components.ui_query_input import render_query_input
from components.ui_results import render_results, render_empty_state
from components.search_engine import execute_search


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MemeMatcher | Meme Retrieval System",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# INITIALIZATION
# =========================
# Initialize session state
initialize_state()

# Inject custom styles
inject_styles()

# Load lightweight resources immediately (fast - just JSON/pickle files)
memes, memes_by_id = load_data()
bm25_index = load_bm25_index()
index_usage, index_visual, sbert_meme_ids = load_faiss_indices()
clip_index, clip_meme_ids = load_clip_index()

# NOTE: SBERT and CLIP models are loaded LAZILY (only when needed for search)
# This dramatically speeds up initial page load


# =========================
# MAIN UI
# =========================
st.title("MemeMatcher")
st.markdown("*Advanced meme search with BM25, SBERT, and CLIP*")
st.markdown("---")

# Configuration bar (horizontal)
config = render_config_bar(len(memes))
mode = config["mode"]
top_k = config["top_k"]
w_bm25 = config["w_bm25"]
w_sbert = config["w_sbert"]

# Query input
query_text, query_image, search_clicked = render_query_input(mode)

# If history was selected, use that query text and trigger search
if st.session_state.get("selected_history_query"):
    query_text = st.session_state.selected_history_query
    st.session_state.selected_history_query = None  # Clear after use


# =========================
# SEARCH EXECUTION
# =========================
search_triggered = st.session_state.do_search

if search_triggered:
    st.session_state.do_search = False
    
    # Validate input
    if mode == "CLIP (Image)":
        if query_image is None:
            st.warning("Please upload an image")
            st.stop()
    else:
        if not query_text.strip():
            st.warning("Please enter a search query")
            st.stop()
    
    try:
        # Load models LAZILY based on search mode (only load what's needed)
        sbert_model = None
        clip_model = None
        clip_processor = None
        clip_device = None
        
        if mode in ["SBERT", "Hybrid"]:
            sbert_model = get_sbert_model()
        
        if mode == "CLIP (Image)":
            clip_model, clip_processor, clip_device = get_clip_model_and_processor()
        
        # Execute search
        results = execute_search(
            mode=mode,
            query_text=query_text,
            query_image=query_image,
            top_k=top_k,
            w_bm25=w_bm25,
            w_sbert=w_sbert,
            bm25_index=bm25_index,
            sbert_model=sbert_model,
            index_usage=index_usage,
            index_visual=index_visual,
            sbert_meme_ids=sbert_meme_ids,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_index=clip_index,
            clip_meme_ids=clip_meme_ids,
            clip_device=clip_device
        )
        
        # Update state
        st.session_state.last_topk = results
        st.session_state.has_results = True
        st.session_state.last_mode = mode
        
        if mode == "CLIP (Image)":
            st.session_state.last_query_image = query_image
            st.session_state.last_query_text = ""
        else:
            st.session_state.last_query_text = query_text
            st.session_state.last_query_image = None
            add_to_search_history(query_text, mode)
    
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.info("Run `python -m src.build` to create indexes first")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


# =========================
# RENDER RESULTS
# =========================
if st.session_state.has_results:
    render_results(
        results=st.session_state.last_topk,
        memes_by_id=memes_by_id,
        query_text=st.session_state.last_query_text,
        query_image=st.session_state.last_query_image,
        mode=st.session_state.last_mode
    )
else:
    render_empty_state()

