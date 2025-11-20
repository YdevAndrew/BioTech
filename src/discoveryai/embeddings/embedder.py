from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode em batches. Retorna np.ndarray dtype float32 shape (n, dim)
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype="float32")

        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        return embeddings