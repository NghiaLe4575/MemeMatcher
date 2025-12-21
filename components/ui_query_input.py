"""Query input UI component"""
import streamlit as st
from PIL import Image
from components.state import trigger_search


def render_query_input(mode: str):
    """Render query input interface based on search mode
    
    Args:
        mode: Search mode ("BM25", "SBERT", "Hybrid", "CLIP (Image)")
        
    Returns:
        tuple: (query_text, query_image, search_clicked)
    """
    query_image = None
    query_text = ""
    search_clicked = False
    
    # Create search section with styling
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    if mode == "CLIP (Image)":
        st.markdown("### Image Search")
        st.markdown("Upload an image to find similar memes")
        
        # Image upload for CLIP mode
        uploaded = st.file_uploader(
            "Upload an image to search similar memes",
            type=["png", "jpg", "jpeg", "webp"],
            on_change=trigger_search,
            key="image_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded is not None:
            query_image = Image.open(uploaded).convert("RGB")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(query_image, caption="Query image", use_container_width=True)
        elif st.session_state.last_mode == "CLIP (Image)" and st.session_state.last_query_image is not None:
            query_image = st.session_state.last_query_image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(query_image, caption="Last query image", use_container_width=True)
    
    else:
        st.markdown("### Search Query")
        
        # Check if a history item was selected (set before widget instantiation)
        if "selected_history_query" not in st.session_state:
            st.session_state.selected_history_query = None
        
        # Determine default value for text input
        default_value = ""
        if st.session_state.selected_history_query:
            default_value = st.session_state.selected_history_query
        elif st.session_state.last_mode != "CLIP (Image)":
            default_value = st.session_state.last_query_text
        
        # Text input for BM25/SBERT/Hybrid modes
        col1, col2 = st.columns([6, 1])

        with col1:
            query_text = st.text_input(
                "Enter your search query",
                placeholder="Type your search query... e.g., success, awkward moment",
                value=default_value,
                label_visibility="collapsed",
                key="query_text_input",
                on_change=trigger_search
            )

        with col2:
            search_clicked = st.button("Search", use_container_width=True, type="primary", on_click=trigger_search)
        
        # Show recent searches as clickable chips below the search bar
        if st.session_state.search_history:
            st.caption("Recent:")
            history_cols = st.columns(min(len(st.session_state.search_history), 5))
            for idx, hist in enumerate(st.session_state.search_history[:5]):
                with history_cols[idx]:
                    if st.button(
                        hist["query"][:20] + ("..." if len(hist["query"]) > 20 else ""),
                        key=f"hist_{idx}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        # Store in separate state variable, then rerun
                        st.session_state.selected_history_query = hist["query"]
                        st.session_state.do_search = True
                        st.rerun()
    
    # Search button for CLIP mode
    if mode == "CLIP (Image)":
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            search_clicked = st.button("Search Memes", use_container_width=True, type="primary", on_click=trigger_search)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return query_text, query_image, search_clicked
