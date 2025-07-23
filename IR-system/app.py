import streamlit as st
import sys
import os
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

sys.path.append(os.path.join(os.path.dirname(__file__), 'Retrieval'))
from Retrieval.main_3 import SearchEngine

st.set_page_config(
    page_title="Research Paper Search Engine",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_search_engine():
    class Args:
        def __init__(self):
            root = os.path.dirname(__file__)
            self.dataset         = os.path.join(root, "dataset", "arXiv") + os.sep
            self.out_folder      = os.path.join(root, "output") + os.sep
            self.segmenter       = "punkt"
            self.tokenizer       = "ptb"
            self.w2v_model_path  = os.path.join(root, "models", "GoogleNews-vectors-negative300.bin")
            self.grid_search     = False
            self.use_dpr         = False
            self.dpr_top_k       = 20
            self.max_papers      = 10000

    return SearchEngine(Args())

def main():
    st.title("Research Paper Search Engine")
    st.markdown("Search for relevant research papers")

    with st.sidebar:
        st.header("About")
        st.write("This search engine uses:")
        st.write("- **BM25** ranking")
        st.write("- **LSA** (Latent Semantic Analysis)")
        st.write("- **Word2Vec** query expansion")

        st.header("Dataset Info")
        st.write("ArXiv Papers processed from metadata")
        arxiv_docs_path = os.path.join("dataset", "arXiv", "arxiv-metadata-oai-snapshot.json")
        if os.path.exists(arxiv_docs_path):
            st.success("✅ ArXiv data processed and ready")
        else:
            st.warning("⚠️ ArXiv data will be processed on first search")

    st.subheader("Search Interface")
    col1, col2 = st.columns([4,1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., neural networks, transformer models, computer vision"
        )
    with col2:
        top_k = st.selectbox("Results:",[5,10,15,20],index=0)

    search_engine = load_search_engine()

    if st.button("🔍 Search Papers", type="primary", use_container_width=True):
        if not query or not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching for relevant papers... This may take a moment for the first search."):
            try:
                res = search_engine.search_papers(query, top_k=top_k)
                st.success(f"Found {len(res)} relevant papers")
                st.subheader(f"Search results for: '{query}'")

                if not res:
                    st.info("No papers found matching your query. Try different keywords.")
                    return

                for i, paper in enumerate(res, 1):
                    with st.expander(f"**{i}. {paper['title']}**", expanded=(i <= 3)):
                        info_col, link_col = st.columns([3, 1])

                        with info_col:
                            st.write("**Abstract:**")
                            abstract = paper.get('abstract', 'No abstract available')
                            if len(abstract) > 500:
                                abstract = abstract[:500] + "..."
                            st.write(abstract)

                            if paper.get('authors'):
                                st.write("**Authors:**")
                                st.write(", ".join(paper['authors'][:5]))
                                if len(paper['authors']) > 5:
                                    st.write(f"... and {len(paper['authors']) - 5} more")

                            if paper.get('categories'):
                                st.write("**Categories:**")
                                st.write(", ".join(paper['categories'][:5]))

                        with link_col:
                            pid = paper['id']
                            arxiv_url = f"https://arxiv.org/abs/{pid}"
                            pdf_url   = f"https://arxiv.org/pdf/{pid}.pdf"

                            st.markdown(f"**Paper ID:** {pid}")
                            st.markdown(f"[📄 View on ArXiv]({arxiv_url})")
                            st.markdown(f"[📁 Download PDF]({pdf_url})")
                            st.code(f"arXiv:{pid}", language="text")

            except Exception as e:
                st.error(f"Error during search: {e}")
                st.write("Please check:")
                st.write("- Word2Vec model at `models/GoogleNews-vectors-negative300.bin`")
                st.write("- ArXiv JSON at `dataset/arXiv/arxiv-metadata-oai-snapshot.json`")

if __name__ == "__main__":
    main()