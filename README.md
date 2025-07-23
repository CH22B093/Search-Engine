# 📚 Research Paper Search Engine

An interactive search engine to explore research papers from an arXiv metadata snapshot. It combines traditional IR techniques with modern embeddings and optional neural re-ranking.

---

## 🚀 Features

- 🔍 **Retrieval Models:**
  - BM25 (lexical)
  - LSA (Latent Semantic Analysis)
  - Word2Vec-based query expansion
  - DPR (Dense Passage Retrieval, optional re-ranking)

- 🧠 **Interfaces:**
  - Streamlit Web UI
  - Command-Line Interface (CLI)

- ⚡ **Optimizations:**
  - Caches preprocessed data and indices
  - Loads millions of abstracts (with development-time cap)

---

## 🗂️ Project Structure

```
IR-system/
├── app.py                        # Streamlit front-end
├── dataset/
│   └── arXiv/
│       └── arxiv-metadata-oai-snapshot.json
├── models/
│   └── GoogleNews-vectors-negative300.bin
├── output/                       # Intermediate files & index dumps
└── Retrieval/
    ├── main_3.py                # CLI & SearchEngine wrapper
    ├── information_Retrieval_3.py  # BM25, LSA, W2V, DPR logic
    ├── sentenceSegmentation.py
    ├── tokenization.py
    ├── inflectionReduction.py
    ├── stopwordRemoval.py
    └── evaluation.py
```

---

## 🧰 Prerequisites

- Python 3.8 to 3.11 (3.12 may require PyTorch nightly)
- `pip` or `conda`
- Required NLTK data: `"punkt"`, `"stopwords"`

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/IR-system.git
cd IR-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary>📄 Sample <code>requirements.txt</code> includes:</summary>

- streamlit  
- rank_bm25  
- scikit-learn  
- gensim  
- nltk  
- sentence-transformers  
- torch  
- matplotlib  

</details>

### 3. Download NLTK Resources
```bash
python - <<EOF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
EOF
```

### 4. Prepare Data

- Place the arXiv JSON snapshot in:
  ```
  dataset/arXiv/arxiv-metadata-oai-snapshot.json
  ```

- Place Word2Vec binary model in:
  ```
  models/GoogleNews-vectors-negative300.bin
  ```

---

## 🖥️ Usage

### A. Streamlit Web App

```bash
python -m streamlit run app.py
```

- Open browser: [http://localhost:8501](http://localhost:8501)
- Enter your query
- Choose number of top-K results
- View abstracts, authors, categories, and arXiv links

---

### B. Command-Line Interface

Run from project root:

```bash
python Retrieval/main_3.py [--options]
```

#### ⚙️ Available CLI Options

| Argument           | Description                                           |
|--------------------|-------------------------------------------------------|
| `--dataset`        | Path to dataset folder (default: `../dataset/arXiv/`) |
| `--out_folder`     | Path to output folder (default: `../output/`)         |
| `--segmenter`      | Sentence segmenter: `punkt` or `naive`                |
| `--tokenizer`      | Tokenizer: `ptb` or `naive`                           |
| `--w2v_model_path` | Path to Word2Vec binary file                          |
| `--max_papers`     | Max number of papers to load (for dev)                |
| `--use_dpr`        | Enable DPR reranking                                  |
| `--dpr_top_k`      | Top-K results to rerank using DPR                     |
| `--grid_search`    | Run grid search on evaluation set                     |
| `--custom`         | Prompt a custom query for retrieval                   |

#### 🧪 Example Commands

- Evaluate dataset with default settings:
  ```bash
  python Retrieval/main_3.py
  ```

- Run a custom query:
  ```bash
  python Retrieval/main_3.py --custom
  ```

---

## ⚙️ Configuration Notes

- In `app.py` → `load_search_engine()`:
  - `max_papers`: cap number of abstracts loaded (e.g. 50000)
  - `use_dpr`: set to `True` to enable neural reranking

- In `main_3.py`:
  - Extend or adjust CLI defaults in the parser section
