from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, k=5):
        """
        candidates = lista de dicts:
        [
            {
                "text": "...",
                "metadata": {...},
                "distance": float
            }
        ]
        """

        if not candidates:
            return []

        # cria pares (query, texto)
        pairs = [(query, c["text"]) for c in candidates]

        # modelo retorna um score por par
        scores = self.model.predict(pairs)

        # anexa score ao dict
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        # ordena pelo score descrescente
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:k]
