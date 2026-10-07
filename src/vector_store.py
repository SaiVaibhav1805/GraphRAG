import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple

class VectorStore:
    def __init__(self, dimension: int = 384, index_path: str = "data/cache/faiss_index.bin"):
        self.dimension = dimension
        self.index_path = index_path
        self.meta_path = index_path.replace(".bin", "_meta.json")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids: List[str] = []
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Build FAISS index from chunks and their normalized embedding matrix."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunks_by_id = {c["chunk_id"]: c for c in chunks}

    def save(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        meta = {
            "chunk_ids": self.chunk_ids,
            "chunks_by_id": self.chunks_by_id
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved vector index to '{self.index_path}' and metadata to '{self.meta_path}'.")

    def load(self) -> bool:
        """Load FAISS index and metadata from disk if they exist."""
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False
        
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.chunks_by_id = meta["chunks_by_id"]
        print(f"Loaded vector store with {self.index.ntotal} items from '{self.index_path}'.")
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search vector index for nearest top_k chunks.
        Returns list of (chunk_dict, cosine_similarity_score).
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            cid = self.chunk_ids[idx]
            chunk = self.chunks_by_id[cid]
            results.append((chunk, float(score)))

        return results

if __name__ == "__main__":
    vs = VectorStore(dimension=384)
    dummy_embeds = np.random.randn(10, 384).astype(np.float32)
    faiss.normalize_L2(dummy_embeds)
    dummy_chunks = [{"chunk_id": f"c_{i}", "title": f"Doc {i}", "text": f"Content {i}"} for i in range(10)]
    vs.build_index(dummy_chunks, dummy_embeds)
    res = vs.search(dummy_embeds[0], top_k=3)
    print("Search test top result:", res[0][0]["title"], "score:", res[0][1])
