import numpy as np
from discoveryai.vectorstore.faiss_store import FaissStore
from discoveryai.embeddings.embedder import Embedder


class VectorService:
    def __init__(self, dim=384, store_path="data/faiss"):
        """
        Serviços de vetorização + FAISS.
        dim = dimensão do embedding (Seu modelo atual usa 384)
        """
        self.embedder = Embedder()
        self.store = FaissStore(dim)
        self.store_path = store_path

    # ---------------------------------------------------------
    # INDEXAÇÃO COMPLETA
    # chunks esperados no formato:
    # {
    #     "text": "...",
    #     "metadata": {
    #         "source": "...",
    #         "page": int,
    #         "chunk_id": "pX_cY"
    #     }
    # }
    # ---------------------------------------------------------
    def index_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]

        vectors = self.embedder.encode(texts)

        # garantir float32 para FAISS
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        self.store.add(vectors, texts, metadata)
        self.store.save(self.store_path)

        return {
            "chunks_indexed": len(chunks),
            "saved_to": self.store_path
        }

    # ---------------------------------------------------------
    # CONSULTA SEMÂNTICA
    # ---------------------------------------------------------
    def semantic_search(self, query, k=5, filters=None):
        vec = self.embedder.encode([query])

        # garantir float32
        if vec.dtype != np.float32:
            vec = vec.astype("float32")

        return self.store.search(vec, k=k, filters=filters)

    # ---------------------------------------------------------
    # RECARREGAR ARMAZENAMENTO FAISS
    # ---------------------------------------------------------
    def load_store(self):
        self.store = FaissStore.load(self.store_path)
