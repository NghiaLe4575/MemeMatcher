"""Configuration constants for MemeMatcher application"""

# Data paths
DATA_PATH = "data/memes.json"
ARTIFACTS_DIR = "artifacts"
OUTPUT_ROOT = "outputs"

# BM25 configuration
BM25_INDEX_PATH = f"{ARTIFACTS_DIR}/bm25_index.pkl"

# SBERT configuration
SBERT_USAGE_INDEX_PATH = f"{ARTIFACTS_DIR}/sbert_usage.index"
SBERT_VISUAL_INDEX_PATH = f"{ARTIFACTS_DIR}/sbert_visual.index"
SBERT_MEME_IDS_PATH = f"{ARTIFACTS_DIR}/sbert_meme_ids.json"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# CLIP configuration
CLIP_INDEX_PATH = f"{ARTIFACTS_DIR}/clip_images.index"
CLIP_MEME_IDS_PATH = f"{ARTIFACTS_DIR}/clip_meme_ids.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Search modes
SEARCH_MODES = ["BM25", "SBERT", "Hybrid", "CLIP (Image)"]

# Default weights for hybrid search
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_SBERT_WEIGHT = 0.6

# UI configuration
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
RESULTS_PER_ROW = 3
USAGE_PREVIEW_LENGTH = 200
