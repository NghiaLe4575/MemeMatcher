"""Search engine for MemeMatcher - executes queries across all retrieval methods"""
from PIL import Image
from src.normalize import normalize_text
from src.retrieval_bm25 import search as bm25_search
from src.retrieval_sbert import search_faiss_sbert
from src.retrieval_clip import search_faiss_clip_image
from src.fusion import normalize, fuse


def execute_search(
    mode: str,
    query_text: str,
    query_image: Image.Image,
    top_k: int,
    w_bm25: float,
    w_sbert: float,
    bm25_index,
    sbert_model,
    index_usage,
    index_visual,
    sbert_meme_ids,
    clip_model,
    clip_processor,
    clip_index,
    clip_meme_ids,
    clip_device
):
    """Execute search query based on selected mode
    
    Args:
        mode: Search mode ("BM25", "SBERT", "Hybrid", "CLIP (Image)")
        query_text: Text query string
        query_image: PIL Image for CLIP search
        top_k: Number of results to return
        w_bm25: Weight for BM25 in hybrid mode
        w_sbert: Weight for SBERT in hybrid mode
        bm25_index: Loaded BM25 index
        sbert_model: Loaded SBERT model
        index_usage: FAISS index for usage text
        index_visual: FAISS index for visual description
        sbert_meme_ids: Meme ID mapping for SBERT
        clip_model: CLIP model
        clip_processor: CLIP processor
        clip_index: FAISS index for CLIP
        clip_meme_ids: Meme ID mapping for CLIP
        clip_device: Device for CLIP (cuda/cpu)
        
    Returns:
        list: Search results with format [{"meme_id": str, "score": float, "winner_field": str|None}, ...]
    """
    mode_lower = mode.lower()
    
    # Normalize query text for BM25/SBERT/Hybrid
    if mode != "CLIP (Image)":
        q_norm = normalize_text(query_text)
    
    bm25_scores = {}
    sbert_scores = {}
    sbert_by_id = {}
    
    # Execute retrieval based on mode
    if mode_lower in ("bm25", "hybrid"):
        bm25_raw = bm25_search(q_norm, bm25_index)
        bm25_scores = normalize({r["meme_id"]: r["score"] for r in bm25_raw})
    
    if mode_lower in ("sbert", "hybrid"):
        sbert_raw = search_faiss_sbert(
            query_text,
            model=sbert_model,
            index_usage=index_usage,
            index_visual=index_visual,
            meme_ids=sbert_meme_ids
        )
        sbert_by_id = {r["meme_id"]: r for r in sbert_raw}
        sbert_scores = normalize({r["meme_id"]: r["score"] for r in sbert_raw})
    
    if mode == "CLIP (Image)":
        clip_raw = search_faiss_clip_image(
            query_pil_image=query_image,
            model=clip_model,
            processor=clip_processor,
            index=clip_index,
            meme_ids=clip_meme_ids,
            top_n=100,
            device=clip_device
        )
        # Normalize to 0..1
        clip_scores = normalize({r["meme_id"]: r["score"] for r in clip_raw})
    
    # Fusion logic
    if mode_lower == "bm25":
        final_scores = bm25_scores
    elif mode_lower == "sbert":
        final_scores = sbert_scores
    elif mode == "CLIP (Image)":
        final_scores = clip_scores
    else:  # Hybrid
        final_scores = fuse(bm25_scores, sbert_scores, w_bm25=w_bm25, w_sbert=w_sbert)
    
    # Rank and take top_k
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Format results
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
    
    return topk
