import os
from datetime import datetime
import warnings
import pickle
import json
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data_loader import load_memes
from src.normalize import normalize_text
from src.retrieval_bm25 import search as bm25_search
from src.retrieval_sbert import search as sbert_search
from src.fusion import normalize, fuse
from src.export_images import export
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"

BM25_INDEX_PATH = f"{ARTIFACTS_DIR}/bm25_index.pkl"
SBERT_EMB_PATH = f"{ARTIFACTS_DIR}/sbert_embeddings.json"

QUERY_TEXT = "third world success kid"
MODE = "hybrid" # bm25 | sbert | hybrid
TOP_K = 20

EXPORT_IMAGES = True
OUTPUT_ROOT = "outputs"


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# =========================


def main():
    # Load data and model
    memes = load_memes(DATA_PATH)
    memes_by_id = {m["meme_id"]: m for m in memes}
    model = SentenceTransformer(MODEL_NAME)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_index= pickle.load(f)
    with open(SBERT_EMB_PATH, "r", encoding="utf-8") as f:
        sbert_embeddings = json.load(f)
        
    print("Data and model loaded successfully")

    # Normalize query (BM25 only)
    q_norm = normalize_text(QUERY_TEXT)

    bm25_scores = {}
    sbert_scores = {}
    sbert_by_id = {}

    # --- Retrieval ---
    if MODE in ("bm25", "hybrid"):
        bm25_raw = bm25_search(q_norm, BM25_INDEX_PATH, bm25_index)
        bm25_scores = normalize({
            r["meme_id"]: r["score"] for r in bm25_raw
        })

    if MODE in ("sbert", "hybrid"):
        sbert_raw = sbert_search(QUERY_TEXT, SBERT_EMB_PATH, model, sbert_embeddings)
        sbert_by_id = {r["meme_id"]: r for r in sbert_raw}
        sbert_scores = normalize({
            r["meme_id"]: r["score"] for r in sbert_raw
        })

    # --- Fusion ---
    if MODE == "bm25":
        final_scores = bm25_scores
    elif MODE == "sbert":
        final_scores = sbert_scores
    else:
        final_scores = fuse(bm25_scores, sbert_scores)

    # --- Ranking ---
    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K]

    # --- Result shaping ---
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

    # --- Print results ---
    print(f"\nQuery: {QUERY_TEXT}")
    print(f"Mode: {MODE}\n")

    for i, r in enumerate(topk, 1):
        print(
            f"{i}. {r['meme_id']}  "
            f"score={r['score']:.4f}  "
            f"field={r.get('winner_field')}"
        )


if __name__ == "__main__":
    main()
