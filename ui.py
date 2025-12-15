import streamlit as st
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data_loader import load_memes
from src.normalize import normalize_text
from src.retrieval_bm25 import search as bm25_search
from src.retrieval_sbert import search as sbert_search
from src.fusion import normalize, fuse
from src.export_images import export

# =========================
# CONFIG
# =========================
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"
BM25_INDEX_PATH = f"{ARTIFACTS_DIR}/bm25_index.pkl"
SBERT_EMB_PATH = f"{ARTIFACTS_DIR}/sbert_embeddings.json"
OUTPUT_ROOT = "outputs"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MemeMatcher | Meme Retrieval System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS - MODERN WHITE THEME
# =========================
st.markdown("""
    <style>
        /* Global font and background */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background-color: #FFFFFF;
            color: #132338;
        }

        /* Smooth rounded corners */
        .stTextInput input, .stSelectbox div, .stButton button, .stSlider {
            border-radius: 10px !important;
            transition: all 0.3s ease;
        }

        /* Modern button styling */
        .stButton button {
            background: linear-gradient(135deg, #475C78 0%, #8D7D5F 100%);
            color: #E2D2B3;
            font-weight: 700;
            border: none;
            box-shadow: 0 6px 14px rgba(19, 35, 56, 0.25);
            padding: 0.85rem 2.5rem;
            font-size: 1rem;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 18px rgba(19, 35, 56, 0.3);
            background: linear-gradient(135deg, #132338 0%, #475C78 100%);
        }

        /* Config panel cards */
        .config-card {
            background: #F5F0E6;
            border: 2px solid #E2D2B3;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 8px rgba(19, 35, 56, 0.08);
        }

        /* Header styling */
        h1 { color: #132338; font-weight: 800; letter-spacing: -1px; }
        h2, h3 { color: #475C78; font-weight: 700; }

        /* Card styling for results */
        .meme-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(19, 35, 56, 0.12);
            transition: all 0.3s ease;
            border: 2px solid #9AA7B8;
        }
        .meme-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(19, 35, 56, 0.18);
            border-color: #FFA600;
        }

        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 700;
            margin: 0.25rem;
        }
        .badge-usage { background-color: #9AA7B8; color: #132338; }
        .badge-visual { background-color: #E2D2B3; color: #475C78; }
        .badge-field { background-color: #FFA600; color: #132338; }

        /* Context/usage text styling */
        .usage-context {
            background-color: #F5F0E6;
            border-left: 4px solid #FFA600;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.95rem;
            color: #475C78;
            margin: 0.75rem 0;
            line-height: 1.5;
        }

        /* Input styling */
        .stTextInput input {
            border: 2px solid #9AA7B8;
            font-size: 1.1rem;
            padding: 0.85rem;
            color: #132338;
        }
        .stTextInput input:focus {
            border-color: #FFA600;
            box-shadow: 0 0 0 3px rgba(255, 166, 0, 0.18);
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data():
    """Load memes data and create lookup dictionary"""
    memes = load_memes(DATA_PATH)
    memes_by_id = {m["meme_id"]: m for m in memes}
    return memes, memes_by_id

# =========================
# SIDEBAR - CONFIGURATION
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # Mode Selection
    st.markdown("**Retrieval Mode**")
    mode = st.radio(
        "Select search algorithm",
        ["BM25", "SBERT", "Hybrid"],
        help="BM25: Keyword-based | SBERT: Semantic | Hybrid: Combined"
    )
    mode_lower = mode.lower()
    
    st.markdown("---")
    
    # Top-K Selection
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="How many memes to retrieve"
    )
    
    # Fusion Weights (only for Hybrid mode)
    if mode == "Hybrid":
        st.markdown("---")
        st.markdown("**Fusion Weights**")
        w_bm25 = st.slider(
            "BM25 Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Weight for keyword matching"
        )
        w_sbert = st.slider(
            "SBERT Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Weight for semantic matching"
        )
        
        # Normalize weights
        total = w_bm25 + w_sbert
        if total > 0:
            w_bm25 = w_bm25 / total
            w_sbert = w_sbert / total
    else:
        w_bm25, w_sbert = 0.4, 0.6
    
    st.markdown("---")
    
    # Export Options
    export_images = st.checkbox(
        "Export Images",
        value=False,
        help="Save top results to outputs folder"
    )
    
    st.markdown("---")
    
    # Stats
    try:
        memes, _ = load_data()
        st.metric("Total Memes", len(memes))
    except:
        st.metric("Total Memes", "N/A")

# =========================
# MAIN INTERFACE
# =========================
st.title("🎯 MemeMatcher")
st.markdown("### Find the perfect meme template for any context")

# Configuration Badge
config_badge = f"🔍 **{mode}** | 📊 Top {top_k}"
if mode == "Hybrid":
    config_badge += f" | ⚖️ {w_bm25:.1f}/{w_sbert:.1f}"
st.markdown(config_badge)

st.markdown("---")

# Search Interface
query_text = st.text_input(
    "Enter your search query",
    placeholder="e.g., 'success', 'awkward moment', 'third world success kid'",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_button = st.button("🔍 Search Memes", width='stretch')

st.markdown("---")

# =========================
# QUERY EXECUTION & RESULTS
# =========================
if search_button:
    if not query_text.strip():
        st.warning("⚠️ Please enter a search query")
    else:
        try:
            # Load data
            memes, memes_by_id = load_data()
            
            # Loading spinner with mode-specific message
            spinner_messages = {
                "bm25": "🔤 Searching BM25 index...",
                "sbert": "🧠 Computing semantic embeddings...",
                "hybrid": "⚡ Running hybrid search (BM25 + SBERT)..."
            }
            
            with st.spinner(spinner_messages[mode_lower]):
                # Normalize query
                q_norm = normalize_text(query_text)
                
                bm25_scores = {}
                sbert_scores = {}
                sbert_by_id = {}
                
                # Retrieval
                if mode_lower in ("bm25", "hybrid"):
                    bm25_raw = bm25_search(q_norm, BM25_INDEX_PATH)
                    bm25_scores = normalize({
                        r["meme_id"]: r["score"] for r in bm25_raw
                    })
                
                if mode_lower in ("sbert", "hybrid"):
                    sbert_raw = sbert_search(query_text, SBERT_EMB_PATH)
                    sbert_by_id = {r["meme_id"]: r for r in sbert_raw}
                    sbert_scores = normalize({
                        r["meme_id"]: r["score"] for r in sbert_raw
                    })
                
                # Fusion
                if mode_lower == "bm25":
                    final_scores = bm25_scores
                elif mode_lower == "sbert":
                    final_scores = sbert_scores
                else:
                    final_scores = fuse(bm25_scores, sbert_scores, w_bm25, w_sbert)
                
                # Ranking
                ranked = sorted(
                    final_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                
                # Result shaping
                topk = []
                for meme_id, score in ranked:
                    winner_field = None
                    if sbert_by_id:
                        winner_field = sbert_by_id.get(meme_id, {}).get("winner_field")
                    
                    topk.append({
                        "meme_id": meme_id,
                        "score": score,
                        "winner_field": winner_field
                    })
                
                # Export if enabled
                if export_images and topk:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = os.path.join(OUTPUT_ROOT, f"run_{ts}")
                    export(topk, memes_by_id, out_dir)
                    st.success(f"✅ Images exported to `{out_dir}`")
            
            # Display Results
            if not topk:
                st.info("No results found. Try a different query.")
            else:
                st.markdown(f"### 🎯 Top {len(topk)} Results")
                st.markdown(f"*Query: \"{query_text}\"*")
                st.markdown("---")
                
                # Display in grid (2-3 per row based on count)
                cols_per_row = 3 if len(topk) >= 3 else 2
                
                for i in range(0, len(topk), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(topk):
                            result = topk[i + j]
                            meme = memes_by_id[result["meme_id"]]
                            
                            with col:
                                st.markdown(f"**#{i + j + 1}**")
                                
                                # Display image
                                if meme.get("image_paths"):
                                    img_path = meme["image_paths"][0]
                                    if os.path.exists(img_path):
                                        st.image(img_path, width='stretch')
                                    else:
                                        st.warning("Image not found")
                                
                                # Meme name
                                st.markdown(f"**{meme['name']}**")
                                
                                # Score with progress bar
                                st.progress(result["score"])
                                st.caption(f"Score: {result['score']:.4f}")
                                
                                # Winner field badge (SBERT only)
                                if result.get("winner_field"):
                                    badge_class = "badge-usage" if result["winner_field"] == "usage" else "badge-visual"
                                    st.markdown(
                                        f'<span class="badge {badge_class}">{result["winner_field"]}</span>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Use Context / Usage Text
                                if meme.get("text", {}).get("usage_text"):
                                    usage = meme["text"]["usage_text"]
                                    if len(usage) > 200:
                                        usage = usage[:200] + "..."
                                    st.markdown(
                                        f'<div class="usage-context"><strong>📋 Use Case:</strong><br>{usage}</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Meme ID
                                st.caption(f"`{result['meme_id']}`")
        
        except FileNotFoundError as e:
            st.error(f"❌ Missing file: {e}")
            st.info("💡 Run `python -m src.build` to create indexes first")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

else:
    # Empty state
    st.info("👆 Enter a query and click 'Search Memes' to begin")
    st.markdown("#### 💡 Example Queries")
    st.markdown("""
    - `success kid`
    - `awkward moment`
    - `feeling proud but poor`
    - `drake approve disapprove`
    - `distracted boyfriend`
    """)