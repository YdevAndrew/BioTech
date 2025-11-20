import faiss
import numpy as np
import json
import os


class FaissStore:
    def __init__(self, dim):
        """
        dim: dimensionalidade dos embeddings
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []

    # ----------------------------------------------------
    # Adicionar vetores ao índice
    # ----------------------------------------------------
    def add(self, vectors, texts, metadata=None):
        """
        vectors: lista ou numpy array de embeddings
        texts: lista de textos correspondentes
        metadata: lista de dicts (opcional)
        """
        vectors = np.array(vectors).astype("float32")

        if metadata is None:
            metadata = [{} for _ in texts]

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    # ----------------------------------------------------
    # Busca semântica + filtros
    # ----------------------------------------------------
    def search(self, query_vec, k=5, filters=None):
        """
        query_vec: embedding (lista ou array)
        k: quantidade de resultados
        filters: dict opcional (ex: {"tipo": "paper"})
        """
        query_vec = np.array(query_vec).reshape(1, -1).astype("float32")

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            item_meta = self.metadata[idx]

            # aplicar filtros se existirem
            if filters:
                matched = True
                for key, val in filters.items():
                    if item_meta.get(key) != val:
                        matched = False
                        break
                if not matched:
                    continue

            results.append({
                "text": self.texts[idx],
                "distance": float(dist),
                "metadata": item_meta
            })

        return results

    # ----------------------------------------------------
    # Persistência em disco
    # ----------------------------------------------------
    def save(self, path):
        """
        Salva o índice FAISS + textos + metadados
        """
        os.makedirs(path, exist_ok=True)

        # salva índice
        faiss.write_index(self.index, f"{path}/index.faiss")

        # salva textos e metadata
        with open(f"{path}/store.json", "w", encoding="utf8") as f:
            json.dump({
                "dim": self.dim,
                "texts": self.texts,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------
    # Carregar repositório salvo
    # ----------------------------------------------------
    @classmethod
    def load(cls, path):
        """
        Carrega um índice FAISS + store.json
        """
        index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/store.json", "r", encoding="utf8") as f:
            data = json.load(f)

        store = cls(data["dim"])
        store.index = index
        store.texts = data["texts"]
        store.metadata = data["metadata"]

        return store
