import streamlit as st
import os
import pickle
import json
import warnings
import html
warnings.filterwarnings("ignore", category=FutureWarning)

import faiss
from sentence_transformers import SentenceTransformer

from src.data_loader import load_memes
from src.normalize import normalize_text
from src.retrieval_bm25 import search as bm25_search
from src.retrieval_sbert import search_faiss_sbert
from src.fusion import normalize, fuse
from src.export_images import export
import html

# =========================
# CONFIG
# =========================
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"

BM25_INDEX_PATH = f"{ARTIFACTS_DIR}/bm25_index.pkl"

SBERT_USAGE_INDEX_PATH  = f"{ARTIFACTS_DIR}/sbert_usage.index"
SBERT_VISUAL_INDEX_PATH = f"{ARTIFACTS_DIR}/sbert_visual.index"
SBERT_MEME_IDS_PATH     = f"{ARTIFACTS_DIR}/sbert_meme_ids.json"

OUTPUT_ROOT = "outputs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# =========================

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
# LOAD DATA / MODELS / INDICES (CACHED, LOADED ONCE)
# =========================
@st.cache_data
def load_data():
    memes = load_memes(DATA_PATH)
    memes_by_id = {m["meme_id"]: m for m in memes}
    return memes, memes_by_id

@st.cache_resource
def get_sbert_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_bm25_index():
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_faiss_indices():
    idx_usage = faiss.read_index(SBERT_USAGE_INDEX_PATH)
    idx_visual = faiss.read_index(SBERT_VISUAL_INDEX_PATH)
    with open(SBERT_MEME_IDS_PATH, "r", encoding="utf-8") as f:
        meme_ids = json.load(f)
    return idx_usage, idx_visual, meme_ids

# Force-load at app startup (first render)
model = get_sbert_model()
memes, memes_by_id = load_data()
bm25_index = load_bm25_index()
index_usage, index_visual, sbert_meme_ids = load_faiss_indices()

# Global search state and variables
if "do_search" not in st.session_state:
    st.session_state.do_search = False
    
def do_search():
    if st.session_state.get("query_text", "").strip():
        st.session_state.do_search = True
        
if "w_bm25" not in st.session_state:
    st.session_state.w_bm25 = 0.4
if "w_sbert" not in st.session_state:
    st.session_state.w_sbert = 0.6
    
if "last_topk" not in st.session_state:
    st.session_state.last_topk = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "has_results" not in st.session_state:
    st.session_state.has_results = False

    
# =========================
# Helper
# =========================
@st.dialog("Detail")
def show_full_usage(title: str, text: str):
    st.markdown(f"**{title}**")
    st.write(text)


# =========================
# SIDEBAR - CONFIGURATION
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Retrieval Mode**")
    mode = st.radio(
        "Select search algorithm",
        ["BM25", "SBERT", "Hybrid"],
        help="BM25: Keyword-based | SBERT: Semantic | Hybrid: Combined",
        label_visibility="collapsed",
        on_change=do_search
    )
    mode_lower = mode.lower()

    st.markdown("---")

    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        on_change=do_search
    )

    st.markdown("---")
    st.markdown("**Fusion Weights**")
    # ALWAYS render sliders so state persists; just disable unless Hybrid
    is_hybrid = (mode == "Hybrid")
    w_bm25 = st.slider(
        "BM25 Weight",
        0.0, 1.0, 1.0 - st.session_state.w_sbert,
        step=0.05,
        key="w_bm25",
        on_change=do_search,
        disabled=not is_hybrid
    )

    w_sbert = st.slider(
        "SBERT Weight",
        0.0, 1.0, 1.0 - st.session_state.w_bm25,
        step=0.05,
        key="w_sbert",
        on_change=do_search,
        disabled=not is_hybrid
    )
        
    total = w_bm25 + w_sbert
    if total > 0:
        w_bm25 /= total
        w_sbert /= total

    #st.markdown("---")
    #export_images = st.checkbox("Export Images", value=False)

    st.markdown("---")
    st.metric("Total Memes", len(memes))

# =========================
# MAIN INTERFACE
# =========================
st.title("🎯 MemeMatcher")

config_badge = f"🔍 **{mode}** | 📊 Top {top_k}"
if mode == "Hybrid":
    config_badge += f" | ⚖️ {w_bm25:.2f}/{w_sbert:.2f}"
st.markdown(config_badge)
st.markdown("---")

# Search box
query_text = st.text_input(
    "Enter your search query",
    placeholder="e.g., 'success', 'awkward moment'",
    label_visibility="collapsed",
    key = "query_text",
    on_change=do_search
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_button = st.button("🔍 Search Memes", use_container_width=True, on_click=do_search)

st.markdown("---")

# =========================
# QUERY EXECUTION & RESULTS
# =========================
search_triggered = st.session_state.do_search 


if search_triggered:
    st.session_state.do_search = False
    if not query_text.strip():
        st.warning("⚠️ Please enter a search query")
    else:
        try:
            # No spinner needed
            # spinner_messages = {
            #     "bm25": "🔤 Searching BM25 index...",
            #     "sbert": "🧠 Searching SBERT (FAISS)...",
            #     "hybrid": "⚡ Running hybrid search (BM25 + SBERT via FAISS)..."
            # }

            # with st.spinner(spinner_messages[mode_lower]):
                q_norm = normalize_text(query_text)

                bm25_scores = {}
                sbert_scores = {}
                sbert_by_id = {}

                # --- Retrieval
                if mode_lower in ("bm25", "hybrid"):
                    bm25_raw = bm25_search(q_norm, bm25_index)
                    bm25_scores = normalize({r["meme_id"]: r["score"] for r in bm25_raw})

                if mode_lower in ("sbert", "hybrid"):
                    sbert_raw = search_faiss_sbert(
                        query_text,
                        model=model,
                        index_usage=index_usage,
                        index_visual=index_visual,
                        meme_ids=sbert_meme_ids
                    )
                    sbert_by_id = {r["meme_id"]: r for r in sbert_raw}
                    sbert_scores = normalize({r["meme_id"]: r["score"] for r in sbert_raw})

                # --- Fusion ---
                if mode_lower == "bm25":
                    final_scores = bm25_scores
                elif mode_lower == "sbert":
                    final_scores = sbert_scores
                else:
                    final_scores = fuse(bm25_scores, sbert_scores, w_bm25=w_bm25, w_sbert=w_sbert)

                # --- Rank (cut only at end) ---
                ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

                # --- Shape results ---
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
                st.session_state.last_topk = topk
                st.session_state.last_query = query_text
                st.session_state.has_results = True


                # Export if enabled
                # if export_images and topk:
                #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                #     out_dir = os.path.join(OUTPUT_ROOT, f"run_{ts}")
                #     export(topk, memes_by_id, out_dir)
                #     st.success(f"✅ Images exported to `{out_dir}`")

        except FileNotFoundError as e:
            st.error(f"❌ Missing file: {e}")
            st.info("💡 Run `python -m src.build` to create indexes first")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)
            
            
# =========================
# RENDER PERSISTED RESULTS
# =========================
topk = st.session_state.last_topk
persisted_query = st.session_state.last_query

if st.session_state.has_results:
    if not topk:
        st.info("No results found. Try a different query.")
    else:
        st.markdown(f"### 🎯 Top {len(topk)} Results")
        st.markdown(f"*Query: \"{persisted_query}\"*")
        st.markdown("---")

        cols_per_row = 3 if len(topk) >= 3 else 2

        for i in range(0, len(topk), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j >= len(topk):
                    continue

                result = topk[i + j]
                meme = memes_by_id[result["meme_id"]]

                with col:
                    st.markdown(f"**#{i + j + 1}**")

                    if meme.get("image_paths"):
                        img_path = meme["image_paths"][0]
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.warning("Image not found")

                    st.markdown(f"**{meme['name']}**")

                    st.progress(float(result["score"]))
                    st.caption(f"Score: {result['score']:.4f}")

                    if result.get("winner_field"):
                        badge_class = "badge-usage" if result["winner_field"] == "usage" else "badge-visual"
                        st.markdown(
                            f'<span class="badge {badge_class}">{result["winner_field"]}</span>',
                            unsafe_allow_html=True
                        )

                    usage_full = meme.get("text", {}).get("usage_text", "")
                    if usage_full:
                        preview_len = 200
                        is_long = len(usage_full) > preview_len
                        usage_preview = usage_full[:preview_len] + "..." if is_long else usage_full

                        st.markdown(
                            f'<div class="usage-context"><strong>📋 Use Case:</strong><br>{html.escape(usage_preview)}</div>',
                            unsafe_allow_html=True
                        )
                        if is_long:
                            st.button(
                                "Show more",
                                key=f"more_{result['meme_id']}",
                                on_click=show_full_usage,
                                args=(meme["name"], usage_full)
                            )

                    st.caption(f"`{result['meme_id']}`")
else:
    st.info("👆 Enter a query and click 'Search Memes' to begin")
    st.markdown("#### 💡 Example Queries")
    st.markdown("""
    - `success kid`
    - `awkward moment`
    - `feeling proud but poor`
    - `drake approve disapprove`
    - `distracted boyfriend`
    """)
