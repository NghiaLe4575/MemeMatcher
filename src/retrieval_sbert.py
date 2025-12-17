import json
import numpy as np

def cosine(a, b):
    return float(np.dot(a, b))

def search(query, emb_path, model):

    q = model.encode(query, normalize_embeddings=True)

    with open(emb_path, "r", encoding="utf-8") as f:
        emb = json.load(f)

    results = []

    for meme_id in emb["usage"]:
        su = cosine(q, emb["usage"][meme_id])
        sv = cosine(q, emb["visual"][meme_id])
        score = max(su, sv)
        winner = "usage" if su >= sv else "visual"

        results.append({
            "meme_id": meme_id,
            "score": score,
            "winner_field": winner,
            "sim_usage": su,
            "sim_visual": sv
        })

    return results  
