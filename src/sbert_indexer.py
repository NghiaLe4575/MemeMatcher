import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_embeddings(memes, out_path):
    model = SentenceTransformer(MODEL_NAME)

    usage_emb = {}
    visual_emb = {}

    for m in tqdm(memes, desc="Embedding memes"):
        meme_id = m["meme_id"]
        usage_emb[meme_id] = model.encode(
            m["text"]["usage_text"], normalize_embeddings=True
        ).tolist()
        visual_emb[meme_id] = model.encode(
            m["text"]["visual_description"], normalize_embeddings=True
        ).tolist()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "usage": usage_emb,
            "visual": visual_emb
        }, f)

