from rank_bm25 import BM25Okapi
import re

def tokenize(text: str):
    # Extrai apenas palavras, ignora pontuação
    return re.findall(r"\b\w+\b", text.lower())

class BM25Search:
    def __init__(self):
        self.docs = []
        self.tokens = []
        self.bm25 = None

    def add(self, texts):
        self.docs.extend(texts)
        tokenized = [tokenize(t) for t in texts]
        self.tokens.extend(tokenized)
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query, k=5):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # top-k
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        return [
            {"text": self.docs[idx], "score": float(score)}
            for idx, score in top
        ]
