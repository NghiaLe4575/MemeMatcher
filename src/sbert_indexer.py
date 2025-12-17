import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_sbert_faiss(memes, out_dir: str):
    """
    Outputs:
      - {out_dir}/sbert_meme_ids.json
      - {out_dir}/sbert_usage.npy
      - {out_dir}/sbert_visual.npy
      - {out_dir}/sbert_usage.index
      - {out_dir}/sbert_visual.index
    """
    model = SentenceTransformer(MODEL_NAME)

    meme_ids = []
    usage_vecs = []
    visual_vecs = []

    for m in tqdm(memes, desc="Embedding memes (SBERT)"):
        meme_id = m["meme_id"]
        meme_ids.append(meme_id)

        u = model.encode(m["text"]["usage_text"], normalize_embeddings=True)
        v = model.encode(m["text"]["visual_description"], normalize_embeddings=True)

        usage_vecs.append(u)
        visual_vecs.append(v)

    usage = np.asarray(usage_vecs, dtype=np.float32)
    visual = np.asarray(visual_vecs, dtype=np.float32)

    # Persist arrays + id order
    with open(f"{out_dir}/sbert_meme_ids.json", "w", encoding="utf-8") as f:
        json.dump(meme_ids, f, ensure_ascii=False, indent=2)

    np.save(f"{out_dir}/sbert_usage.npy", usage)
    np.save(f"{out_dir}/sbert_visual.npy", visual)

    # Build FAISS indices (cosine = inner product because vectors are normalized)
    dim = usage.shape[1]
    index_usage = faiss.IndexFlatIP(dim)
    index_visual = faiss.IndexFlatIP(dim)

    index_usage.add(usage)
    index_visual.add(visual)

    faiss.write_index(index_usage, f"{out_dir}/sbert_usage.index")
    faiss.write_index(index_visual, f"{out_dir}/sbert_visual.index")
