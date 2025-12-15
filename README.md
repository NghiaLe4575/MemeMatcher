# MemeMatcher

## 1. Install dependencies

Create a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
````

---

## 2. Project layout (important folders)

```
data/
├── memes.json        # Meme metadata
└── images/           # Local meme images

artifacts/            # Auto-generated indexes (created by build)
outputs/              # Retrieved images (created by query)

src/
├── build.py          # Build indexes (run once)
└── query.py          # Run queries
```

---

## 3. Build indexes (run once)

From the project root:

```bash
python -m src.build
```

This will generate:

* BM25 index
* SBERT embeddings

in the `artifacts/` folder.

Re-run **only if `memes.json` changes**.

---

## 4. Configure a query

Edit the **CONFIG section** at the top of `src/query.py`:

```python
QUERY_TEXT = "awkward silence"
MODE = "hybrid"     # bm25 | sbert | hybrid
TOP_K = 5

EXPORT_IMAGES = True
OUTPUT_ROOT = "outputs"
```
- `QUERY_TEXT`: the input query
- `MODE`: retrieval mode: `bm25` for keyword-based, `sbert` for embedding based, `hybrid` for combined method.
---

## 5. Run a query

From the project root:

```bash
python -m src.query
```
In the mean time, the results in each query is save in `outputs/`.

---

## 6. Output

### Console

Shows ranked meme IDs, scores, and matched field (`usage` or `visual`).

### Images

If `EXPORT_IMAGES = True`, top results are copied to:

```
outputs/run_YYYYMMDD_HHMMSS/
├── 01.jpg
├── 02.png
└── meta.json
```

Images are renamed by rank.

---

## Notes

* Always run with `python -m src.<file>`
* Do not run scripts directly (`python src/query.py`)
* Build must be run before querying
