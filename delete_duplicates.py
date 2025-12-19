import json

def deduplicate_memes_by_id(
    input_path: str,
    output_path: str,
    keep: str = "first"  # "first" or "last"
):
    """
    Remove duplicate meme_id entries from a JSON list.

    Args:
        input_path: path to input .json file
        output_path: path to output .json file
        keep: "first" or "last" occurrence of duplicate meme_id
    """

    with open(input_path, "r", encoding="utf-8") as f:
        memes = json.load(f)

    seen = {}
    for meme in memes:
        meme_id = meme.get("meme_id")
        if meme_id is None:
            continue  # or raise error if you want strictness

        if keep == "first":
            if meme_id not in seen:
                seen[meme_id] = meme
        elif keep == "last":
            seen[meme_id] = meme
        else:
            raise ValueError("keep must be 'first' or 'last'")

    deduplicated = list(seen.values())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)

    print(f"Original: {len(memes)} memes")
    print(f"After dedup: {len(deduplicated)} memes")

def main():
    deduplicate_memes_by_id(
    input_path="data/memes.json",
    output_path="data/memes_dedup.json",
    keep="first"
)

if __name__ == "__main__":
    main()