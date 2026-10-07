import re
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set

from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline

class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        graph: nx.DiGraph,
        max_neighbors_per_node: int = 10,
        traversal_hops: int = 2,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.embedding = embedding_pipeline
        self.graph = graph
        self.max_neighbors_per_node = max_neighbors_per_node
        self.traversal_hops = traversal_hops
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

        # Index graph node names and embeddings for fast linking
        self.node_names = list(self.graph.nodes())
        if self.node_names:
            self.node_embeddings = self.embedding.encode(self.node_names, show_progress_bar=False)
        else:
            self.node_embeddings = np.array([])

    def identify_question_entities(self, question: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Identify entities in the question and link them to graph nodes (FR-16).
        Combines string exact/substring matching with embedding similarity.
        """
        matched_nodes = []
        q_lower = question.lower()

        # 1. Direct title/string matching
        for node in self.node_names:
            n_lower = node.lower()
            if len(n_lower) > 3 and n_lower in q_lower:
                matched_nodes.append((node, 1.0))

        # 2. Embedding similarity matching for un-matched question phrases
        if len(matched_nodes) < top_n and len(self.node_names) > 0:
            q_vec = self.embedding.encode(question, show_progress_bar=False)
            sims = np.dot(self.node_embeddings, q_vec.T).flatten()
            
            top_indices = np.argsort(sims)[::-1][:top_n]
            for idx in top_indices:
                node = self.node_names[idx]
                score = float(sims[idx])
                if score >= 0.55 and node not in [m[0] for m in matched_nodes]:
                    matched_nodes.append((node, score))

        return matched_nodes[:top_n]

    def traverse_graph(
        self,
        start_nodes: List[str],
        max_hops: int = 2
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """
        Traverse 1 to 2 hops from start nodes, capped at max_neighbors_per_node (FR-17, FR-18).
        Returns:
            chunk_min_hops: dict mapping chunk_id -> minimum hop distance from start nodes
            triples: list of traversed sub-graph triple dicts
        """
        chunk_min_hops: Dict[str, int] = {}
        traversed_triples = []
        visited_nodes: Set[str] = set()

        for start_node in start_nodes:
            if not self.graph.has_node(start_node):
                continue

            # Level 0 (start node itself if it has an attached document chunk)
            node_data = self.graph.nodes[start_node]
            if node_data.get("is_document") and "chunk_id" in node_data:
                chunk_min_hops[node_data["chunk_id"]] = 0

            # BFS Traversal
            current_level = [start_node]
            visited_nodes.add(start_node)

            for hop in range(1, max_hops + 1):
                next_level = []
                for u in current_level:
                    # Get neighbors (out-edges and in-edges)
                    neighbors = list(self.graph.successors(u)) + list(self.graph.predecessors(u))
                    
                    # Cap neighbors per node to avoid hub explosion
                    if len(neighbors) > self.max_neighbors_per_node:
                        # Prioritize neighbors with higher degree or title match
                        neighbors = neighbors[:self.max_neighbors_per_node]

                    for v in neighbors:
                        # Collect edge metadata & chunk IDs
                        if self.graph.has_edge(u, v):
                            edge_data = self.graph[u][v]
                            for cid in edge_data.get("chunk_ids", []):
                                if cid not in chunk_min_hops or hop < chunk_min_hops[cid]:
                                    chunk_min_hops[cid] = hop
                            for rel in edge_data.get("relations", ["related to"]):
                                traversed_triples.append({"subject": u, "relation": rel, "object": v})

                        if self.graph.has_edge(v, u):
                            edge_data = self.graph[v][u]
                            for cid in edge_data.get("chunk_ids", []):
                                if cid not in chunk_min_hops or hop < chunk_min_hops[cid]:
                                    chunk_min_hops[cid] = hop
                            for rel in edge_data.get("relations", ["related to"]):
                                traversed_triples.append({"subject": v, "relation": rel, "object": u})

                        if v not in visited_nodes:
                            visited_nodes.add(v)
                            next_level.append(v)

                current_level = next_level

        return chunk_min_hops, traversed_triples

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "hybrid",  # "baseline", "graph_only", "hybrid"
        max_hops: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve top_k chunks for a question.
        Returns:
            reranked_chunks: List of top_k chunk dicts with scores
            subgraph_triples: List of traversed triples
            debug_info: metadata dictionary
        """
        # 1. Vector Search
        q_vec = self.embedding.encode(question, show_progress_bar=False)
        vector_results = self.vector_store.search(q_vec, top_k=top_k * 2)
        vec_chunk_scores = {c["chunk_id"]: sim for c, sim in vector_results}

        if mode == "baseline":
            top_chunks = [c for c, _ in vector_results[:top_k]]
            return top_chunks, [], {"mode": "baseline", "vec_count": len(top_chunks)}

        # 2. Graph Traversal
        matched_entities = self.identify_question_entities(question)
        start_nodes = [node for node, score in matched_entities]
        chunk_hops, subgraph_triples = self.traverse_graph(start_nodes, max_hops=max_hops)

        if mode == "graph_only":
            # Rank chunks purely by graph hop distance and edge frequency
            graph_chunk_list = []
            for cid, hop in chunk_hops.items():
                if cid in self.vector_store.chunks_by_id:
                    chunk = self.vector_store.chunks_by_id[cid]
                    score = 1.0 / (1.0 + hop)
                    graph_chunk_list.append((chunk, score))
            graph_chunk_list.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [c for c, _ in graph_chunk_list[:top_k]]
            return top_chunks, subgraph_triples, {"mode": "graph_only", "matched_entities": start_nodes}

        # 3. Hybrid Reranking (FR-19)
        all_candidate_ids = set(vec_chunk_scores.keys()).union(set(chunk_hops.keys()))

        scored_candidates = []
        for cid in all_candidate_ids:
            if cid not in self.vector_store.chunks_by_id:
                continue

            chunk = self.vector_store.chunks_by_id[cid]
            
            # Vector score (normalized 0 to 1)
            v_score = vec_chunk_scores.get(cid, 0.0)
            
            # Graph score based on hop distance
            hop = chunk_hops.get(cid, None)
            if hop is None:
                g_score = 0.0
            elif hop == 0:
                g_score = 1.0
            elif hop == 1:
                g_score = 0.75
            else:
                g_score = 0.50

            combined_score = (self.vector_weight * v_score) + (self.graph_weight * g_score)
            scored_candidates.append((chunk, combined_score, v_score, g_score))

        # Sort by combined score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_reranked_chunks = [c for c, score, v_s, g_s in scored_candidates[:top_k]]

        debug_info = {
            "mode": "hybrid",
            "matched_entities": start_nodes,
            "vector_candidates_count": len(vec_chunk_scores),
            "graph_candidates_count": len(chunk_hops),
            "traversed_triples_count": len(subgraph_triples)
        }

        return top_reranked_chunks, subgraph_triples[:15], debug_info
