from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name


    def embed_texts(self, texts: List[str]):
        """
        TODO:
          - Encode with normalize_embeddings=True
          - Return np.ndarray [n, dim] float32
        """
        embeddings = self.model.encode(
          texts,
          normalize_embeddings=True,
          convert_to_numpy=True,
          batch_size=32,
          show_progress_bar=True,
        )
        embeddings = embeddings.astype(np.float32, copy=False)
        return embeddings