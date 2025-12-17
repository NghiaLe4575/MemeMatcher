import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data_loader import load_memes
from src.bm25_indexer import build_corpus, build_index
from src.sbert_indexer import build_sbert_faiss

# =========================
# CONFIG
# =========================
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"

BM25_CORPUS_PATH = f"{ARTIFACTS_DIR}/bm25_corpus.json"
BM25_INDEX_PATH  = f"{ARTIFACTS_DIR}/bm25_index.pkl"
# SBERT+FAISS outputs are written into ARTIFACTS_DIR
# =========================

def main():
    print("Loading memes...")
    memes = load_memes(DATA_PATH)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Building BM25 corpus...")
    corpus = build_corpus(memes, BM25_CORPUS_PATH)

    print("Building BM25 index...")
    build_index(corpus, BM25_INDEX_PATH)

    print("Building SBERT embeddings + FAISS indices...")
    build_sbert_faiss(memes, ARTIFACTS_DIR)

    print("Build complete.")
    print("Artifacts written to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
