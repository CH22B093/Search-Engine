import os
import json
import argparse
from sys import version_info
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from information_Retrieval_3 import InformationRetrieval
from evaluation import Evaluation

# Python2/3 input() fix
if version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass

class SearchEngine:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        self.informationRetriever = InformationRetrieval(self.args.w2v_model_path)
        self.evaluator = Evaluation()
        self._load_and_index()

    def _load_and_index(self):
        snap_file = os.path.join(self.args.dataset, "arxiv-metadata-oai-snapshot.json")
        max_p = getattr(self.args, "max_papers", None)

        self.docs_json = []
        self.doc_ids   = []
        raw_bodies     = []

        with open(snap_file, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                if max_p and i >= max_p:
                    break
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue

                body = f"{p.get('title','').strip()}. {p.get('abstract','').strip()}"
                authors = [
                    f"{a[1]} {a[0]}"
                    for a in p.get('authors_parsed', [])
                    if isinstance(a,list) and len(a) >= 2
                ]
                categories = p.get('categories','').split()

                doc = {
                    "id":p.get("id",""),
                    "title":p.get("title","").strip(),
                    "body":body,
                    "abstract":p.get("abstract","").strip(),
                    "authors":authors,
                    "categories":categories
                }
                self.docs_json.append(doc)
                self.doc_ids.append(doc["id"])
                raw_bodies.append(body)

        processed = self.preprocessDocs(raw_bodies)
        self.informationRetriever.buildIndex(processed, self.doc_ids)

    def segmentSentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, tokens):
        return self.inflectionReducer.reduce(tokens)

    def removeStopwords(self, tokens):
        return self.stopwordRemover.fromList(tokens)

    def preprocessQueries(self, queries):
        seg = [self.segmentSentences(q) for q in queries]
        tok = [self.tokenize(s)         for s in seg]
        red = [self.reduceInflection(t) for t in tok]
        clean = [self.removeStopwords(r)  for r in red]
        return clean

    def preprocessDocs(self, docs):
        seg = [self.segmentSentences(d) for d in docs]
        tok = [self.tokenize(s)         for s in seg]
        red = [self.reduceInflection(t) for t in tok]
        clean = [self.removeStopwords(r)  for r in red]
        return clean

    def search_papers(self,query,top_k=5):
        """
        Return top_k docs for a single query string.
        """
        proc_q = self.preprocessQueries([query])[0]
        ranked_ids = self.informationRetriever.rank([proc_q],top_n=top_k)[0][0]
        ranked_ids = ranked_ids[:top_k]
        results = []
        for doc_id in ranked_ids:
            doc = next((d for d in self.docs_json if d["id"] == doc_id), None)
            if doc:
                results.append(doc)
        return results
    def handleCustomQuery(self):
        """
        CLI mode: ask for a single query on the console.
        """
        print("Enter query below:")
        q = input().strip()
        res = self.search_papers(q,top_k=5)
        for i, paper in enumerate(res, 1):
            print(f"{i}. {paper['title']}  →  https://arxiv.org/abs/{paper['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main_3.py CLI')
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), os.pardir,"dataset","arXiv") + os.sep,
        help="Path to folder containing arxiv-metadata-oai-snapshot.json"
    )
    parser.add_argument(
        "--out_folder",
        default=os.path.join(os.path.dirname(__file__), os.pardir,"output") + os.sep,
        help="Where to write intermediate files"
    )
    parser.add_argument(
        "--segmenter", default="punkt",
        help="Sentence segmenter: [naive|punkt]"
    )
    parser.add_argument(
        "--tokenizer", default="ptb",
        help="Tokenizer: [naive|ptb]"
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Run custom query on CLI instead of Evaluate"
    )
    parser.add_argument(
        "--w2v_model_path",
        default=os.path.join(os.path.dirname(__file__), os.pardir, "models", "GoogleNews-vectors-negative300.bin"),
        help="Path to pretrained Word2Vec binary"
    )
    parser.add_argument(
        "--max_papers", type=int, default=2000,
        help="Cap how many lines of the ArXiv snapshot to load (dev speed-up)"
    )
    parser.add_argument(
        "--grid_search", action="store_true",
        help="Perform grid-search on Cranfield eval"
    )
    parser.add_argument(
        "--use_dpr", action="store_true",
        help="Enable DPR reranking (requires torch & transformers)"
    )
    parser.add_argument(
        "--dpr_top_k", type=int, default=20,
        help="How many docs to rerank with DPR"
    )

    args = parser.parse_args()
    engine = SearchEngine(args)

    if args.custom:
        engine.handleCustomQuery()