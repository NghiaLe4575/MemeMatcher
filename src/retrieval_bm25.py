import pickle
import numpy as np

def search(query, index_path, data, top_n=20):
    
    bm25 = data["bm25"]
    meme_ids = data["meme_ids"]

    scores = bm25.get_scores(query.split())
    idxs = np.argsort(scores)[::-1][:top_n]

    return [
        {"meme_id": meme_ids[i], "score": float(scores[i])}
        for i in idxs
    ]
