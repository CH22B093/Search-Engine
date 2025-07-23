======================== Research Paper Search Engine
Overview
This application provides an interactive web interface and a command-line mode for searching research papers from an arXiv metadata snapshot. It supports: • Traditional IR ranking (BM25) • Latent Semantic Analysis (LSA) • Word2Vec query expansion • Optional DPR (Dense Passage Retrieval) reranking

The front-end is built with Streamlit (app.py). The core search engine logic lives in Retrieval/main_3.py and Retrieval/information_Retrieval_3.py.

Features
• Load and preprocess millions of arXiv abstracts on startup (capped for development). • Build a combined BM25 + LSA + Word2Vec index once per session. • Fast query ranking with optional DPR re-ranking. • Streamlit UI: enter a query, select top-K results, view abstracts, authors, categories, and open papers on arXiv. • CLI mode: evaluate dataset (Cranfield), perform grid-search, or issue custom queries from the console. • Caching with Streamlit to avoid repeated index builds.

Project Structure
IR-system/ ├── app.py ← Streamlit front-end
├── dataset/
│ └── arXiv/
│ └── arxiv-metadata-oai-snapshot.json
├── models/
│ └── GoogleNews-vectors-negative300.bin
├── output/ ← Intermediate files & serialized index dumps
└── Retrieval/
├── main_3.py ← CLI entry point & SearchEngine wrapper
├── information_Retrieval_3.py← Core IR functionality (BM25, LSA, W2V, DPR)
├── sentenceSegmentation.py
├── tokenization.py
├── inflectionReduction.py
├── stopwordRemoval.py
└── evaluation.py 

Prerequisites
• Python 3.8+ (tested up to 3.11; 3.12 may require nightly PyTorch)
• pip or conda environment
• NLTK data (“punkt”, “stopwords”)




Install dependencies:
pip install -r requirements.txt
(Sample requirements.txt includes:
streamlit
rank_bm25
scikit-learn
gensim
nltk
sentence-transformers
torch
matplotlib
)

Download NLTK data:
python - <<EOF
import nltk;
nltk.download('punkt');
nltk.download('stopwords');
EOF

Data Preparation

Placed the arXiv snapshot line-delimited JSON in:
dataset/arXiv/arxiv-metadata-oai-snapshot.json

Ensure “models/GoogleNews-vectors-negative300.bin” is present.

Usage

A. Streamlit Web App

From project root:
python -m streamlit run app.py
Browse to http://localhost:8501
Enter your query, choose number of results, click “Search Papers”.
Expand each result to view abstract, authors, categories, and links to arXiv.
B. Command-Line Interface

Run CLI mode (evaluate dataset or custom query):
python Retrieval/main_3.py [--options]
Available arguments (all optional):
--dataset path to dataset folder (default: ../dataset/arXiv/)
--out_folder path for output files (default: ../output/)
--segmenter [punkt|naive]
--tokenizer [ptb|naive]
--w2v_model_path path to Word2Vec binary
--max_papers integer cap for number of papers to load
--use_dpr flag to enable DPR reranking
--dpr_top_k top-k for DPR
--grid_search perform grid-search on evaluation set
--custom prompt for a single custom query
Examples:
• Evaluate dataset with default settings:
python Retrieval/main_3.py
• Custom query mode:
python Retrieval/main_3.py --custom

Configuration
• In app.py → load_search_engine() → Args:
• max_papers: number of lines to load at startup (None for all)
• use_dpr: set to True to enable DPR re-ranking (slow due to encoding)
• In main_3.py → CLI parser: to adjust defaults or add new flags.

