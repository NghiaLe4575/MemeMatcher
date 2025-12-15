# score normalization
def normalize(scores):
    if not scores:
        return {}

    vals = [v for v in scores.values()]
    lo, hi = min(vals), max(vals)

    if hi == lo:
        return {k: 0.0 for k in scores}

    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def fuse(bm25, sbert, w_bm25=0.4, w_sbert=0.6):
    all_ids = set(bm25) | set(sbert)
    out = {}

    for mid in all_ids:
        out[mid] = (
            w_bm25 * bm25.get(mid, 0.0) +
            w_sbert * sbert.get(mid, 0.0)
        )

    return out
