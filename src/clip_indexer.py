import os
import json
import numpy as np
import faiss
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def _load_image(path: str):
    # Always convert to RGB for CLIP
    img = Image.open(path).convert("RGB")
    return img

@torch.inference_mode()
def build_clip_faiss(memes, out_dir: str, device: str | None = None):
    """
    Builds image-image CLIP embeddings for each meme (uses meme["image_paths"][0]).
    Outputs:
      - {out_dir}/clip_meme_ids.json
      - {out_dir}/clip_images.npy
      - {out_dir}/clip_images.index
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    meme_ids = []
    vecs = []

    for m in tqdm(memes, desc="Embedding memes (CLIP image->image)"):
        meme_id = m["meme_id"]
        img_path = m["image_paths"][0]

        if not os.path.exists(img_path):
            # Skip missing images, but keep going
            continue

        img = _load_image(img_path)
        inputs = processor(images=img, return_tensors="pt").to(device)

        feats = model.get_image_features(**inputs)   # (1, dim)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize -> cosine via inner product
        emb = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

        meme_ids.append(meme_id)
        vecs.append(emb)

    if not vecs:
        raise RuntimeError("No CLIP embeddings were built (all images missing?)")

    embs = np.stack(vecs, axis=0).astype(np.float32)  # (N, dim)

    # Persist arrays + id order
    with open(f"{out_dir}/clip_meme_ids.json", "w", encoding="utf-8") as f:
        json.dump(meme_ids, f, ensure_ascii=False, indent=2)
    np.save(f"{out_dir}/clip_images.npy", embs)

    # FAISS index (cosine similarity == inner product because vectors normalized)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, f"{out_dir}/clip_images.index")
