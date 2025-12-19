import json
import re

def load_memes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        memes = json.load(f)

    seen_ids = set()

    for m in memes:
        meme_id = m.get("meme_id")
        if not meme_id: # or not re.fullmatch(r"[a-z0-9_-%]+", meme_id):
            raise ValueError(f"Invalid meme_id: {meme_id}")

        if meme_id in seen_ids:
            raise ValueError(f"Duplicate meme_id: {meme_id}")
        seen_ids.add(meme_id)

        if not m.get("image_paths") or len(m["image_paths"]) == 0:
            raise ValueError(f"{meme_id}: image_paths missing")

        text = m.get("text", {})
        vd = text.get("visual_description", "").strip()
        ut = text.get("usage_text", "").strip()

        if not vd or not ut:
            raise ValueError(f"{meme_id}: text fields missing")

    return memes
