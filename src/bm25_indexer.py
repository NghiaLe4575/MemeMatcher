import json
import pickle
from rank_bm25 import BM25Okapi
from src.normalize import normalize_text

def build_corpus(memes, out_path):
    corpus = []
    for m in memes:
        text = (
            m["text"]["visual_description"] + " " +
            m["text"]["usage_text"]
        )
        corpus.append({
            "meme_id": m["meme_id"],
            "text": normalize_text(text)
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    return corpus

def build_index(corpus, out_path):
    tokenized = [c["text"].split() for c in corpus]
    bm25 = BM25Okapi(tokenized)

    with open(out_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "meme_ids": [c["meme_id"] for c in corpus]
        }, f)
