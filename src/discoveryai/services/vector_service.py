import numpy as np
from discoveryai.vectorstore.faiss_store import FaissStore
from discoveryai.embeddings.embedder import Embedder
from discoveryai.nlp.reranker import Reranker
from discoveryai.generation.response_generator import ResponseGenerator




class VectorService:
    def __init__(self, dim=768, store_path="data/faiss"):
        self.embedder = Embedder()
        self.store = FaissStore(dim)
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
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

    def ask(self, query, k=5, rerank_k=20):
        """
        Pipeline completo de RAG:
        FAISS -> Reranking -> Geração de resposta
        """

        # 1. busca vetorial + reranking
        context = self.semantic_search(query, k=k, rerank_k=rerank_k)

        # 2. gerar resposta
        answer = self.generator.generate(query, context)

        return {
            "query": query,
            "answer": answer,
            "context": context
        }

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
    def semantic_search(self, query, k=5, rerank_k=20):
        """
        k = resultados finais
        rerank_k = quantos pegar do FAISS antes do rerank
        """
        vec = self.embedder.embed([query])

        # busca bruta primeiro
        faiss_results = self.store.search(vec, k=rerank_k)

        # reranking com cross-encoder
        ranked = self.reranker.rerank(query, faiss_results, k=k)

        return ranked


    # ---------------------------------------------------------
    # RECARREGAR ARMAZENAMENTO FAISS
    # ---------------------------------------------------------
    def load_store(self):
        self.store = FaissStore.load(self.store_path)
