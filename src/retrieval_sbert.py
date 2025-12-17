import numpy as np

def search_faiss_sbert(query: str, model, index_usage, index_visual, meme_ids, top_n: int = 100):
    """
    Returns: list of dicts for candidate union:
      {meme_id, score, winner_field, sim_usage, sim_visual}
    score = max(sim_usage, sim_visual)
    """
    q = model.encode(query, normalize_embeddings=True).astype(np.float32).reshape(1, -1)

    # Search each index
    Du, Iu = index_usage.search(q, top_n)
    Dv, Iv = index_visual.search(q, top_n)

    # FAISS returns arrays shape (1, top_n)
    Du, Iu = Du[0], Iu[0]
    Dv, Iv = Dv[0], Iv[0]

    # Build maps idx -> score
    usage_scores = {}
    visual_scores = {}

    for idx, score in zip(Iu, Du):
        if idx < 0:
            continue
        usage_scores[int(idx)] = float(score)

    for idx, score in zip(Iv, Dv):
        if idx < 0:
            continue
        visual_scores[int(idx)] = float(score)

    # Union candidates
    all_idxs = set(usage_scores.keys()) | set(visual_scores.keys())

    results = []
    for idx in all_idxs:
        su = usage_scores.get(idx, float("-inf"))
        sv = visual_scores.get(idx, float("-inf"))

        # If only one side exists, the other is -inf; max works.
        score = su if su >= sv else sv
        winner = "usage" if su >= sv else "visual"

        results.append({
            "meme_id": meme_ids[idx],
            "score": score,
            "winner_field": winner,
            "sim_usage": None if su == float("-inf") else su,
            "sim_visual": None if sv == float("-inf") else sv,
        })

    # Sort by semantic score desc (still only candidates)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
