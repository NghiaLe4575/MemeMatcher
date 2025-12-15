import pandas as pd
import json
import re
import os
import requests
import time

INPUT_CSV = 'data/imkg_final_final_final_processor.csv'
OUTPUT_JSON = 'data/memes.json'
IMAGE_DIR = 'data/images'

def to_snake_case(name):
    """Converts 'Bad Luck Brian' to 'bad_luck_brian'"""
    s = str(name).strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_-]+', '_', s)
    return s

def get_first_k_sentences(text, k=2):
    """Extracts the first k sentences from a text block."""
    if pd.isna(text): return ""
    # Split by sentence delimiters
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:k])

def download_image(url, save_path):
    """Downloads image with a user-agent to avoid 403s."""
    if os.path.exists(save_path):
        return True
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Filter for rows with complete essential text data
    # (We drop rows where 'about' or descriptions are missing to ensure high quality)
    df_clean = df.dropna(subset=['about', 'description', 'caption_style_explanation', 'template_url', 'template_title'])
    
    # 2. Sort by popularity and take top 100
    df_clean = df_clean.sort_values(by='total_views', ascending=False).head(100)
    print(f"Selected top {len(df_clean)} high-quality memes.")

    # Prepare output directory
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    memes_list = []
    
    print("Processing and downloading images...")
    for idx, row in df_clean.iterrows():
        meme_id = to_snake_case(row['template_title'])
        image_filename = f"{meme_id}.jpg"
        image_path_rel = f"{IMAGE_DIR}/{image_filename}"
        
        # Data Transformation
        visual_desc = get_first_k_sentences(row['description'], 2)
        usage_txt = row['about'].replace('\n', ' ').strip()
        
        meme_obj = {
            "meme_id": meme_id,
            "name": row['template_title'],
            "image_paths": [image_path_rel],
            "text": {
                "visual_description": visual_desc,
                "usage_text": usage_txt,
            }
        }
        memes_list.append(meme_obj)
        
        # Download Image
        download_image(row['template_url'], image_path_rel)
        
    # Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(memes_list, f, indent=2)
        
    print(f"Successfully created {OUTPUT_JSON} with {len(memes_list)} memes.")

if __name__ == "__main__":
    main()