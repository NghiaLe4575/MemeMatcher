import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def _ensure_rgb(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("RGB")

@torch.inference_mode()
def search_faiss_clip_image(
    query_pil_image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    index,
    meme_ids,
    top_n: int = 100,
    device: str | None = None,
):
    """
    Returns list of dicts:
      {meme_id, score}
    score is cosine similarity (via inner product on normalized vectors).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = _ensure_rgb(query_pil_image)
    inputs = processor(images=img, return_tensors="pt").to(device)

    feats = model.get_image_features(**inputs)  # (1, dim)
    feats = feats / feats.norm(dim=-1, keepdim=True)

    q = feats.detach().cpu().numpy().astype(np.float32)  # (1, dim)

    D, I = index.search(q, top_n)
    D, I = D[0], I[0]

    out = []
    for idx, score in zip(I, D):
        if idx < 0:
            continue
        out.append({"meme_id": meme_ids[int(idx)], "score": float(score)})

    return out
