# MemeMatcher UI - Component Architecture

## Overview

The MemeMatcher UI has been refactored into a modular, component-based architecture for better maintainability and extensibility.

## Structure

```
ui.py                     # Main orchestrator (144 lines)
components/
├── __init__.py           # Package initialization
├── config.py             # Constants and configuration
├── loaders.py            # Cached resource loaders
├── search_engine.py      # Search execution logic
├── state.py              # Session state management
├── styles.py             # CSS styling
├── ui_config_bar.py      # Configuration toolbar (horizontal)
├── ui_query_input.py     # Query input component
└── ui_results.py         # Results display component
```

## Components

### config.py
- All constants (paths, model names, defaults)
- No Streamlit dependencies
- Single source of truth for configuration

### loaders.py
- Cached resource loaders (@st.cache_resource, @st.cache_data)
- Functions: load_data, load_bm25_index, load_faiss_indices, load_clip_index, get_sbert_model, get_clip_model_and_processor
- All resources loaded once at startup

### search_engine.py
- `execute_search()` function handles all search modes
- Mode routing: BM25, SBERT, Hybrid, CLIP
- Score normalization and fusion
- Returns formatted results

### state.py
- Session state initialization
- Helper functions: trigger_search, clear_results, add_to_search_history
- Manages: search flags, weights, results, query history

### styles.py
- `inject_styles()` injects custom CSS
- Modern white theme with gradient buttons
- No emojis in styling

### ui_config_bar.py
- `render_config_bar()` renders horizontal configuration toolbar
- Replaces sidebar with cleaner horizontal layout
- Mode selection via radio buttons
- Advanced options in expander for hybrid weights

### ui_query_input.py
- `render_query_input()` handles text and image input
- Conditional rendering based on mode
- Search history dropdown for quick access
- Triggers search on input change

### ui_results.py
- `render_results()` displays search results in grid
- Interactive filtering by field and score threshold
- Image lightbox for full-size viewing
- Expandable usage text with "Show more"
- `render_empty_state()` shows example queries

## Running the Application

### Option 1: PowerShell (Windows)
```powershell
.\run_ui.ps1
```

### Option 2: Bash (Linux/Mac)
```bash
chmod +x run_ui.sh
./run_ui.sh
```

### Option 3: Manual
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Run Streamlit
streamlit run ui.py
```

## Key Features

### Interactive Features
1. **Result Filtering** - Filter by field (Usage/Visual) and minimum score
2. **Search History** - Quick access to last 10 queries
3. **Image Lightbox** - Click "View Full Size" to see images in modal
4. **Expandable Text** - Long usage text can be expanded with "Show more"

### FAISS Integration
- All indices use FAISS for efficient similarity search
- CLIP: clip_images.index for image-to-image search
- SBERT: sbert_usage.index and sbert_visual.index for semantic text search
- BM25: Traditional keyword search with fusion support

## Search Modes

### BM25
Keyword-based lexical search using rank_bm25 library

### SBERT
Semantic search using Sentence Transformers with dual FAISS indices (usage + visual descriptions)

### Hybrid
Weighted fusion of BM25 and SBERT scores with configurable weights

### CLIP (Image)
Image-to-image search using CLIP embeddings and FAISS
