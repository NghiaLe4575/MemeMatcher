"""Results display UI component"""
import streamlit as st
import html
import os
from pathlib import Path
from PIL import Image
from components.config import RESULTS_PER_ROW, USAGE_PREVIEW_LENGTH


@st.cache_data
def create_thumbnail(img_path: str, max_size: int = 800):
    """Create a thumbnail version of image for faster grid rendering
    
    Args:
        img_path: Path to the image file
        max_size: Maximum dimension (width or height) for thumbnail
        
    Returns:
        PIL.Image or None: Resized image object, or None if failed
    """
    try:
        # Try multiple path resolution strategies
        paths_to_try = [
            img_path,
            os.path.abspath(img_path),
            str(Path(img_path))
        ]
        
        for path in paths_to_try:
            if os.path.isfile(path):
                img = Image.open(path)
                
                # Convert to RGB if necessary (handles RGBA, P, L modes)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Only resize if image is larger than max_size
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                return img
        
        return None
    
    except Exception as e:
        return None


@st.dialog("Meme Details")
def show_full_usage(title: str, text: str):
    """Display full usage text in a modal dialog
    
    Args:
        title: Meme name
        text: Full usage text
    """
    st.markdown(f"**{title}**")
    st.write(text)


@st.dialog("Image Viewer")
def show_image_lightbox(image_path: str, meme_name: str):
    """Display full-size image in a modal dialog
    
    Args:
        image_path: Path to the image file
        meme_name: Name of the meme
    """
    if os.path.exists(image_path):
        st.image(image_path, caption=meme_name, use_container_width=True)
    else:
        st.error("Image not found")


def render_results(results: list, memes_by_id: dict, query_text: str, query_image, mode: str):
    """Render search results in a grid layout
    
    Args:
        results: List of search results [{"meme_id": str, "score": float, "winner_field": str|None}, ...]
        memes_by_id: Dictionary mapping meme_id to meme data
        query_text: The search query text (if applicable)
        query_image: The search query image (if applicable)
        mode: The search mode used
    """
    if not results:
        st.info("No results found. Try a different query.")
        return
    
    # Results header
    st.markdown(f"### Top {len(results)} Results")
    if mode == "CLIP (Image)":
        st.markdown("*Query: image-based search*")
    else:
        st.markdown(f'*Query: "{query_text}"*')
    
    # Result filtering
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        filter_options = ["All", "Usage", "Visual"]
        filter_field = st.selectbox(
            "Filter by field",
            filter_options,
            key="filter_field_select"
        )
    
    with col2:
        score_threshold = st.slider(
            "Minimum score",
            0.0, 1.0,
            value=0.0,
            step=0.05,
            key="score_threshold_slider"
        )
    
    # Apply filters
    filtered_results = results
    if filter_field != "All":
        filter_value = filter_field.lower()
        filtered_results = [r for r in results if r.get("winner_field") == filter_value]
    
    filtered_results = [r for r in filtered_results if r["score"] >= score_threshold]
    
    if not filtered_results:
        st.warning("No results match the current filters.")
        return
    
    st.markdown(f"*Showing {len(filtered_results)} of {len(results)} results*")
    st.markdown("---")
    
    # Render results in grid
    cols_per_row = RESULTS_PER_ROW if len(filtered_results) >= RESULTS_PER_ROW else 2
    
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(filtered_results):
                continue
            
            result = filtered_results[i + j]
            meme = memes_by_id[result["meme_id"]]
            
            with col:
                # Rank number
                st.markdown(f"**#{i + j + 1}**")
                
                # Image with click to enlarge - use thumbnail for grid performance
                if meme.get("image_paths"):
                    img_path = meme["image_paths"][0]
                    
                    # Create thumbnail for grid display
                    thumbnail = create_thumbnail(img_path)
                    
                    if thumbnail:
                        try:
                            st.image(thumbnail, use_container_width=True)
                            if st.button(
                                "View Full Size",
                                key=f"view_{result['meme_id']}",
                                use_container_width=True
                            ):
                                show_image_lightbox(img_path, meme["name"])
                        except Exception as e:
                            st.error(f"Error loading image")
                    else:
                        st.warning("Image not available")
                
                # Meme name
                st.markdown(f"**{meme['name']}**")
                
                # Score progress bar
                st.progress(float(result["score"]))
                st.caption(f"Score: {result['score']:.4f}")
                
                # Winner field badge
                if result.get("winner_field"):
                    badge_class = "badge-usage" if result["winner_field"] == "usage" else "badge-visual"
                    st.markdown(
                        f'<span class="badge {badge_class}">{result["winner_field"]}</span>',
                        unsafe_allow_html=True
                    )
                
                # Usage context with preview/expand
                usage_full = meme.get("text", {}).get("usage_text", "")
                if usage_full:
                    is_long = len(usage_full) > USAGE_PREVIEW_LENGTH
                    usage_preview = usage_full[:USAGE_PREVIEW_LENGTH] + "..." if is_long else usage_full
                    
                    st.markdown(
                        f'<div class="usage-context"><strong>Use Case:</strong><br>{html.escape(usage_preview)}</div>',
                        unsafe_allow_html=True
                    )
                    if is_long:
                        if st.button(
                            "Show more",
                            key=f"more_{result['meme_id']}",
                            use_container_width=True
                        ):
                            show_full_usage(meme["name"], usage_full)
                
                # Meme ID
                st.caption(f"`{result['meme_id']}`")


def render_empty_state():
    """Render empty state when no search has been performed"""
    st.info("Enter a query above to begin searching")
    st.markdown("#### Example Queries")
    st.markdown("""
    - `success kid`
    - `awkward moment`
    - `feeling proud but poor`
    - `drake approve disapprove`
    - `distracted boyfriend`
    """)
