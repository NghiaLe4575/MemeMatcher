import os
import shutil
import json

def export(topk, memes_by_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    manifest = []

    for rank, item in enumerate(topk, start=1):
        meme = memes_by_id[item["meme_id"]]
        img_path = meme["image_paths"][0]

        src = img_path
        ext = os.path.splitext(src)[1]
        dst_name = f"{rank:02d}{ext}"
        dst = os.path.join(out_dir, dst_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            dst_name = None

        manifest.append({
            "rank": rank,
            "meme_id": item["meme_id"],
            "score": item["score"],
            "winner_field": item.get("winner_field"),
            "original_image": src,
            "exported_image": dst_name
        })

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)
