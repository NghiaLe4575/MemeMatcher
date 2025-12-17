import os
from datetime import datetime
import warnings
import pickle
import json
import time

warnings.filterwarnings("ignore", category=FutureWarning)

import faiss
from sentence_transformers import SentenceTransformer

from src.data_loader import load_memes
from src.normalize import normalize_text
from src.retrieval_bm25 import search as bm25_search
from src.retrieval_sbert import search_faiss_sbert
from src.fusion import normalize, fuse
from src.export_images import export

# =========================
# CONFIG
# =========================
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"

BM25_INDEX_PATH = f"{ARTIFACTS_DIR}/bm25_index.pkl"

SBERT_USAGE_INDEX_PATH  = f"{ARTIFACTS_DIR}/sbert_usage.index"
SBERT_VISUAL_INDEX_PATH = f"{ARTIFACTS_DIR}/sbert_visual.index"
SBERT_MEME_IDS_PATH     = f"{ARTIFACTS_DIR}/sbert_meme_ids.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUERY_TEXT = "third world success kid"
MODE = "hybrid"   # bm25 | sbert | hybrid
TOP_K = 20

# Option A knobs (candidate sizes)
BM25_TOP_N  = 150
SBERT_TOP_N = 150

# Fusion weights
W_BM25 = 0.4
W_SBERT = 0.6

EXPORT_IMAGES = True
OUTPUT_ROOT = "outputs"
# =========================


def main():
    # Load data
    start = time.perf_counter()

    print('Loading indices')
    memes = load_memes(DATA_PATH)
    memes_by_id = {m["meme_id"]: m for m in memes}

    # Load BM25 index
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_index = pickle.load(f)

    # Load SBERT model + FAISS indices + id order

    index_usage = faiss.read_index(SBERT_USAGE_INDEX_PATH)
    index_visual = faiss.read_index(SBERT_VISUAL_INDEX_PATH)

    with open(SBERT_MEME_IDS_PATH, "r", encoding="utf-8") as f:
        sbert_meme_ids = json.load(f)
        
    end = time.perf_counter()
    print(f"Load data and indices: {end - start:.6f} seconds")
        
    start = time.perf_counter()
    print('Loading model')
    model = SentenceTransformer(MODEL_NAME)
    
    end = time.perf_counter()
    print(f"Load model: {end - start:.6f} seconds")

    q_norm = normalize_text(QUERY_TEXT)

    bm25_scores = {}
    sbert_scores = {}
    sbert_by_id = {}

    # --- Retrieve candidates ---
    if MODE in ("bm25", "hybrid"):
        bm25_raw = bm25_search(q_norm, bm25_index, top_n=BM25_TOP_N)
        bm25_scores = normalize({r["meme_id"]: r["score"] for r in bm25_raw})

    if MODE in ("sbert", "hybrid"):
        sbert_raw = search_faiss_sbert(
            QUERY_TEXT,
            model=model,
            index_usage=index_usage,
            index_visual=index_visual,
            meme_ids=sbert_meme_ids,
            top_n=SBERT_TOP_N
        )
        sbert_by_id = {r["meme_id"]: r for r in sbert_raw}
        sbert_scores = normalize({r["meme_id"]: r["score"] for r in sbert_raw})

    # --- Fusion (union of candidates) ---
    if MODE == "bm25":
        final_scores = bm25_scores
    elif MODE == "sbert":
        final_scores = sbert_scores
    else:
        final_scores = fuse(bm25_scores, sbert_scores, w_bm25=W_BM25, w_sbert=W_SBERT)

    # --- Rank and cut top-K (only here) ---
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

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

    # --- Export images ---
    if EXPORT_IMAGES:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_ROOT, f"run_{ts}")
        export(topk, memes_by_id, out_dir)

    # --- Print ---
    print(f"\nQuery: {QUERY_TEXT}")
    print(f"Mode: {MODE}")
    print(f"Candidates: BM25_TOP_N={BM25_TOP_N}, SBERT_TOP_N={SBERT_TOP_N}\n")

    for i, r in enumerate(topk, 1):
        print(f"{i}. {r['meme_id']}  score={r['score']:.4f}  field={r.get('winner_field')}")


if __name__ == "__main__":
    main()
