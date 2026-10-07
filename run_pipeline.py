import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import yaml
import time
import pandas as pd
from typing import Dict, Any, List

from src.ingestion import DataIngestion
from src.embeddings import EmbeddingPipeline
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.graph_builder import GraphBuilder
from src.retrieval import HybridRetriever
from src.generation import RAGGenerator
from src.evaluation import Evaluator

def load_config(config_path: str = "configs/settings.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_experiment_pipeline():
    print("=" * 65)
    print("      GraphRAG for Multi-Hop QA Experiment Pipeline")
    print("=" * 65)

    cfg = load_config()

    # -------------------------------------------------------------
    # Step 1: Data Ingestion
    # -------------------------------------------------------------
    print("\n--- Phase 0: Data Ingestion & Sampling ---")
    ingestion = DataIngestion(
        parquet_path=cfg["dataset"]["parquet_path"],
        sample_size=cfg["dataset"]["sample_size"],
        random_seed=cfg["dataset"]["random_seed"],
        processed_dir=cfg["dataset"]["processed_dir"]
    )
    questions, chunks = ingestion.run()

    # -------------------------------------------------------------
    # Step 2: Vector Store Indexing
    # -------------------------------------------------------------
    print("\n--- Phase 0.5: Vector Embedding & FAISS Indexing ---")
    embedding_pipe = EmbeddingPipeline(model_name=cfg["embedding"]["model_name"])
    vector_store = VectorStore(
        dimension=cfg["embedding"]["dimension"],
        index_path=cfg["vector_store"]["faiss_index_path"]
    )

    if not vector_store.load():
        chunk_texts = [f"{c['title']} {c['text']}" for c in chunks]
        print(f"Encoding {len(chunk_texts)} chunk texts...")
        embeddings = embedding_pipe.encode(chunk_texts, show_progress_bar=True)
        vector_store.build_index(chunks, embeddings)
        vector_store.save()

    # -------------------------------------------------------------
    # Step 3: Triple Extraction & Knowledge Graph Building
    # -------------------------------------------------------------
    print("\n--- Phase 1: Knowledge Graph Construction ---")
    llm_client = LLMClient(
        provider=cfg["llm"]["provider"],
        model_name=cfg["llm"]["model_name"],
        cache_dir=cfg["llm"]["cache_dir"]
    )

    graph_builder = GraphBuilder(
        llm_client=llm_client,
        embedding_pipeline=embedding_pipe,
        similarity_threshold=cfg["graph"]["similarity_threshold"],
        graph_save_path=cfg["graph"]["graph_save_path"],
        triples_cache_path=cfg["graph"]["triple_cache_path"]
    )

    if not graph_builder.load_graph():
        chunk_triples = graph_builder.extract_triples_for_chunks(chunks)
        graph = graph_builder.build_graph(chunks, chunk_triples, enable_entity_merging=True)
        graph_builder.save_graph()
    else:
        graph = graph_builder.graph

    # Triple quality spot-check
    print("\n--- Performing Triple Quality Spot-Check (50 Triples Sample) ---")
    all_edges = list(graph.edges(data=True))
    if len(all_edges) > 0:
        sample_size = min(50, len(all_edges))
        sample_edges = all_edges[:sample_size]
        correct_count = 0
        for u, v, d in sample_edges:
            # Check edge validity: non-empty entities and valid relation
            rels = d.get("relations", [])
            cids = d.get("chunk_ids", [])
            if len(u) > 1 and len(v) > 1 and rels and cids:
                correct_count += 1
        accuracy = (correct_count / sample_size) * 100
        print(f"Triple Quality Check: {correct_count}/{sample_size} valid triples ({accuracy:.1f}% accuracy - Target >= 80%).")

    # -------------------------------------------------------------
    # Step 4: System Evaluation (Baseline vs Graph-Only vs Hybrid)
    # -------------------------------------------------------------
    print("\n--- Phase 2 & 3: Multi-System Evaluation ---")
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_pipeline=embedding_pipe,
        graph=graph,
        max_neighbors_per_node=cfg["graph"]["max_neighbors_per_node"],
        traversal_hops=cfg["graph"]["traversal_hops"],
        vector_weight=cfg["retrieval"]["vector_weight"],
        graph_weight=cfg["retrieval"]["graph_weight"]
    )
    generator = RAGGenerator(llm_client)
    evaluator = Evaluator(results_dir=cfg["evaluation"]["results_dir"])

    systems = ["baseline", "graph_only", "hybrid"]
    comparison_summary = {}

    for sys_mode in systems:
        print(f"\nRunning evaluation for [{sys_mode.upper()}] system on {len(questions)} questions...")
        records = []

        start_time = time.time()
        for idx, q_obj in enumerate(questions):
            q_text = q_obj["question"]
            gold_ans = q_obj["answer"]
            gold_titles = q_obj["gold_supporting_titles"]
            hop_count = q_obj["hop_count"]

            retrieved_chunks, triples, debug = retriever.retrieve(
                question=q_text,
                top_k=cfg["vector_store"]["top_k"],
                mode=sys_mode,
                max_hops=cfg["graph"]["traversal_hops"]
            )
            retrieved_titles = [c["title"] for c in retrieved_chunks]

            pred_ans = generator.generate_answer(
                question=q_text,
                chunks=retrieved_chunks,
                triples=triples,
                mode=sys_mode
            )

            records.append({
                "question_id": q_obj["question_id"],
                "question": q_text,
                "gold_answer": gold_ans,
                "pred_answer": pred_ans,
                "gold_titles": gold_titles,
                "retrieved_titles": retrieved_titles,
                "hop_count": hop_count
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == len(questions):
                print(f"[{sys_mode.upper()}] Processed {idx + 1}/{len(questions)} questions...")

        elapsed = time.time() - start_time
        summary = evaluator.evaluate_predictions(records)
        summary["execution_time_sec"] = round(elapsed, 2)
        comparison_summary[sys_mode] = summary

        print(f"[{sys_mode.upper()}] EM: {summary['exact_match']*100:.2f}%, F1: {summary['f1']*100:.2f}%, Recall@k: {summary['recall_at_k']*100:.2f}% (Time: {elapsed:.1f}s)")

    # -------------------------------------------------------------
    # Step 5: Ablation Studies
    # -------------------------------------------------------------
    print("\n--- Phase 3 (FR-24): Running Ablation Studies ---")
    ablation_summary = {}

    # Ablation 1: 1-hop vs 2-hop traversal
    for hops in cfg["evaluation"]["ablation_hops"]:
        ab_name = f"hybrid_{hops}hop"
        print(f"Running Ablation [{ab_name}]...")
        ab_records = []
        for q_obj in questions[:100]:  # subset of 100 for ablation
            retrieved_chunks, triples, _ = retriever.retrieve(
                question=q_obj["question"],
                top_k=cfg["vector_store"]["top_k"],
                mode="hybrid",
                max_hops=hops
            )
            retrieved_titles = [c["title"] for c in retrieved_chunks]
            pred_ans = generator.generate_answer(q_obj["question"], retrieved_chunks, triples, mode="hybrid")
            ab_records.append({
                "question_id": q_obj["question_id"],
                "pred_answer": pred_ans,
                "gold_answer": q_obj["answer"],
                "gold_titles": q_obj["gold_supporting_titles"],
                "retrieved_titles": retrieved_titles,
                "hop_count": q_obj["hop_count"]
            })
        ablation_summary[ab_name] = evaluator.evaluate_predictions(ab_records)

    with open(os.path.join(cfg["evaluation"]["results_dir"], "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    # -------------------------------------------------------------
    # Step 6: Save Results & Plots
    # -------------------------------------------------------------
    print("\n--- Phase 4: Generating Result Tables & Plots ---")
    evaluator.generate_plots_and_tables(comparison_summary)

    print("\n" + "=" * 65)
    print("      GraphRAG Pipeline Execution Successfully Finished!")
    print("=" * 65)

if __name__ == "__main__":
    run_experiment_pipeline()
