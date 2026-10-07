import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = True) -> np.ndarray:
        """Embeds text or list of texts into L2-normalized numpy vectors."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

if __name__ == "__main__":
    ep = EmbeddingPipeline()
    vecs = ep.encode(["Hello world", "GraphRAG for multi-hop QA"])
    print(f"Encoded shape: {vecs.shape}")
