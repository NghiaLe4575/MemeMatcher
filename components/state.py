"""Session state management for MemeMatcher application"""
import streamlit as st
from components.config import DEFAULT_BM25_WEIGHT, DEFAULT_SBERT_WEIGHT


def initialize_state():
    """Initialize all session state variables with default values"""
    
    # Search trigger flag
    if "do_search" not in st.session_state:
        st.session_state.do_search = False
    
    # Hybrid mode weights
    if "w_bm25" not in st.session_state:
        st.session_state.w_bm25 = DEFAULT_BM25_WEIGHT
    if "w_sbert" not in st.session_state:
        st.session_state.w_sbert = DEFAULT_SBERT_WEIGHT
    
    # Search results storage
    if "last_topk" not in st.session_state:
        st.session_state.last_topk = []
    if "has_results" not in st.session_state:
        st.session_state.has_results = False
    
    # Query history
    if "last_query_text" not in st.session_state:
        st.session_state.last_query_text = ""
    if "last_query_image" not in st.session_state:
        st.session_state.last_query_image = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None
    
    # Search history (last 10 queries)
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    # Result filtering
    if "filter_field" not in st.session_state:
        st.session_state.filter_field = "All"
    if "score_threshold" not in st.session_state:
        st.session_state.score_threshold = 0.0


def trigger_search():
    """Set flag to trigger search execution"""
    st.session_state.do_search = True


def clear_results():
    """Clear current search results"""
    st.session_state.has_results = False
    st.session_state.last_topk = []


def add_to_search_history(query_text: str, mode: str):
    """Add query to search history (keep last 10)
    
    Args:
        query_text: The search query text
        mode: The search mode used
    """
    if query_text.strip():
        entry = {"query": query_text, "mode": mode}
        # Remove duplicates
        st.session_state.search_history = [
            h for h in st.session_state.search_history 
            if h["query"] != query_text
        ]
        # Add to front
        st.session_state.search_history.insert(0, entry)
        # Keep only last 10
        st.session_state.search_history = st.session_state.search_history[:10]
