"""Configuration toolbar UI component (replaces sidebar)"""
import streamlit as st
from components.config import SEARCH_MODES, DEFAULT_TOP_K, MAX_TOP_K
from components.state import trigger_search, clear_results


def render_config_bar(total_memes: int):
    """Render horizontal configuration toolbar with dark blue styling
    
    Args:
        total_memes: Total number of memes in database
        
    Returns:
        dict: Configuration settings {mode, top_k, w_bm25, w_sbert}
    """
    # Use container with custom class for dark blue background
    st.markdown('<div class="config-panel">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown("**SEARCH MODE**")
        mode = st.radio(
            "Search Mode",
            SEARCH_MODES,
            horizontal=True,
            help="BM25: Keyword search | SBERT: Semantic search | Hybrid: Combined | CLIP: Image-based",
            on_change=clear_results,
            key="search_mode",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**RESULTS**")
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            on_change=trigger_search,
            key="top_k_slider",
            label_visibility="collapsed"
        )
    
    with col3:
        st.metric("Database", total_memes)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced options for Hybrid mode
    w_bm25 = st.session_state.w_bm25
    w_sbert = st.session_state.w_sbert
    
    if mode == "Hybrid":
        with st.expander("ADVANCED: Fusion Weights", expanded=False):
            st.markdown("Adjust how much each search method contributes to the final results")
            col_a, col_b = st.columns(2)
            
            with col_a:
                w_bm25 = st.slider(
                    "BM25 Weight (Keyword)",
                    0.0, 1.0,
                    step=0.05,
                    key="w_bm25",
                    on_change=trigger_search
                )
            
            with col_b:
                w_sbert = st.slider(
                    "SBERT Weight (Semantic)",
                    0.0, 1.0,
                    step=0.05,
                    key="w_sbert",
                    on_change=trigger_search
                )
            
            # Normalize weights
            total = w_bm25 + w_sbert
            if total > 0:
                w_bm25 /= total
                w_sbert /= total
            
            st.info(f"Normalized: BM25={w_bm25:.2%} | SBERT={w_sbert:.2%}")
    
    return {
        "mode": mode,
        "top_k": top_k,
        "w_bm25": w_bm25,
        "w_sbert": w_sbert
    }
