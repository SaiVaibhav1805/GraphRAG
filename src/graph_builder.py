import os
import json
import pickle
import re
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from tqdm import tqdm

from src.llm_client import LLMClient
from src.embeddings import EmbeddingPipeline

class GraphBuilder:
    def __init__(
        self,
        llm_client: LLMClient,
        embedding_pipeline: EmbeddingPipeline,
        similarity_threshold: float = 0.82,
        graph_save_path: str = "data/cache/knowledge_graph.pkl",
        triples_cache_path: str = "data/cache/triples_cache.json"
    ):
        self.llm = llm_client
        self.embedding = embedding_pipeline
        self.similarity_threshold = similarity_threshold
        self.graph_save_path = graph_save_path
        self.triples_cache_path = triples_cache_path
        self.graph = nx.DiGraph()

    def clean_entity(self, entity_str: str) -> str:
        """Clean and canonicalize entity names."""
        clean = re.sub(r'[\"\']', '', entity_str).strip()
        clean = re.sub(r'\s+', ' ', clean)
        return clean

    def extract_triples_for_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
        """Extract triples for all chunks with disk caching (FR-10, FR-11)."""
        cache = {}
        if os.path.exists(self.triples_cache_path):
            with open(self.triples_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

        chunk_triples = {}
        modified = False

        print(f"Extracting triples for {len(chunks)} chunks...")
        for chunk in tqdm(chunks, desc="Triple Extraction"):
            cid = chunk["chunk_id"]
            if cid in cache:
                chunk_triples[cid] = cache[cid]
            else:
                triples = self.llm.extract_triples(chunk["title"], chunk["text"])
                cleaned_triples = []
                for t in triples:
                    s = self.clean_entity(t["subject"])
                    r = t["relation"].strip().lower()
                    o = self.clean_entity(t["object"])
                    if s and r and o and s.lower() != o.lower():
                        cleaned_triples.append({"subject": s, "relation": r, "object": o})
                
                cache[cid] = cleaned_triples
                chunk_triples[cid] = cleaned_triples
                modified = True

        if modified:
            os.makedirs(os.path.dirname(self.triples_cache_path), exist_ok=True)
            with open(self.triples_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

        return chunk_triples

    def merge_near_duplicate_entities(self, all_entities: List[str], enable_merging: bool = True) -> Dict[str, str]:
        """
        Merge near-duplicate entities using embedding similarity (FR-12).
        Returns entity_map mapping raw entity -> canonical entity.
        """
        unique_entities = sorted(list(set(all_entities)))
        entity_map = {ent: ent for ent in unique_entities}

        if not enable_merging or len(unique_entities) < 2:
            return entity_map

        print(f"Merging near-duplicate entities among {len(unique_entities)} unique names...")
        embeddings = self.embedding.encode(unique_entities, show_progress_bar=False)

        # Pairwise cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    ent_i = unique_entities[i]
                    ent_j = unique_entities[j]
                    
                    # Choose shorter or cleaner string as canonical representative
                    canonical = ent_i if len(ent_i) <= len(ent_j) else ent_j
                    
                    # Update mapping
                    curr_canon_i = entity_map[ent_i]
                    curr_canon_j = entity_map[ent_j]
                    
                    final_canon = curr_canon_i if len(curr_canon_i) <= len(curr_canon_j) else curr_canon_j
                    entity_map[ent_i] = final_canon
                    entity_map[ent_j] = final_canon

        num_merged = len(unique_entities) - len(set(entity_map.values()))
        print(f"Entity deduplication complete: {num_merged} duplicate entities merged.")
        return entity_map

    def build_graph(
        self,
        chunks: List[Dict[str, Any]],
        chunk_triples: Dict[str, List[Dict[str, str]]],
        enable_entity_merging: bool = True
    ) -> nx.DiGraph:
        """
        Construct NetworkX Knowledge Graph (FR-13).
        Links every edge to source chunk IDs.
        """
        # Collect all entities
        raw_entities = []
        for triples in chunk_triples.values():
            for t in triples:
                raw_entities.append(t["subject"])
                raw_entities.append(t["object"])

        entity_map = self.merge_near_duplicate_entities(raw_entities, enable_merging=enable_entity_merging)

        self.graph = nx.DiGraph()

        # Add document/paragraph title nodes as primary anchors
        for chunk in chunks:
            title = self.clean_entity(chunk["title"])
            cid = chunk["chunk_id"]
            if not self.graph.has_node(title):
                self.graph.add_node(title, is_document=True, chunk_id=cid)

        # Add extracted triples
        for cid, triples in chunk_triples.items():
            for t in triples:
                subj = entity_map.get(t["subject"], t["subject"])
                obj = entity_map.get(t["object"], t["object"])
                rel = t["relation"]

                if subj == obj:
                    continue

                if not self.graph.has_node(subj):
                    self.graph.add_node(subj, is_document=False)
                if not self.graph.has_node(obj):
                    self.graph.add_node(obj, is_document=False)

                if self.graph.has_edge(subj, obj):
                    # Append chunk ID to edge metadata
                    edge_data = self.graph[subj][obj]
                    if cid not in edge_data["chunk_ids"]:
                        edge_data["chunk_ids"].append(cid)
                    if rel not in edge_data["relations"]:
                        edge_data["relations"].append(rel)
                else:
                    self.graph.add_edge(subj, obj, relations=[rel], chunk_ids=[cid])

        print(f"Graph constructed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def save_graph(self, custom_path: Optional[str] = None):
        """Save graph to disk using pickle (FR-14)."""
        save_path = custom_path or self.graph_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to '{save_path}'.")

    def load_graph(self, custom_path: Optional[str] = None) -> bool:
        """Load graph from disk if present (FR-14)."""
        save_path = custom_path or self.graph_save_path
        if not os.path.exists(save_path):
            return False
        with open(save_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges from '{save_path}'.")
        return True

if __name__ == "__main__":
    llm = LLMClient()
    emb = EmbeddingPipeline()
    gb = GraphBuilder(llm, emb)
    sample_chunks = [
        {"chunk_id": "c1", "title": "Scott Derrickson", "text": "Scott Derrickson is an American film director born in LA."},
        {"chunk_id": "c2", "title": "Doctor Strange", "text": "Doctor Strange is a 2016 film directed by Scott Derrickson."}
    ]
    ct = gb.extract_triples_for_chunks(sample_chunks)
    g = gb.build_graph(sample_chunks, ct)
    print("Nodes:", list(g.nodes()))
    print("Edges:", list(g.edges(data=True)))
