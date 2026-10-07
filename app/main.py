import os
import sys
import yaml
import json
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import DataIngestion
from src.embeddings import EmbeddingPipeline
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.graph_builder import GraphBuilder
from src.retrieval import HybridRetriever
from src.generation import RAGGenerator

st.set_page_config(
    page_title="GraphRAG Multi-Hop QA Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_system_pipeline():
    with open("configs/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    q_path = "data/processed/questions.json"
    c_path = "data/processed/chunks.json"
    if not (os.path.exists(q_path) and os.path.exists(c_path)):
        ingestion = DataIngestion(parquet_path=cfg["dataset"]["parquet_path"], sample_size=cfg["dataset"]["sample_size"])
        questions, chunks = ingestion.run()
    else:
        with open(q_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        with open(c_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    embedding_pipe = EmbeddingPipeline(model_name=cfg["embedding"]["model_name"])
    vector_store = VectorStore(dimension=cfg["embedding"]["dimension"], index_path=cfg["vector_store"]["faiss_index_path"])

    if not vector_store.load():
        chunk_texts = [f"{c['title']} {c['text']}" for c in chunks]
        embeddings = embedding_pipe.encode(chunk_texts, show_progress_bar=False)
        vector_store.build_index(chunks, embeddings)
        vector_store.save()

    llm_client = LLMClient(provider=cfg["llm"]["provider"], model_name=cfg["llm"]["model_name"], cache_dir=cfg["llm"]["cache_dir"])
    graph_builder = GraphBuilder(llm_client, embedding_pipe, similarity_threshold=cfg["graph"]["similarity_threshold"])

    if not graph_builder.load_graph():
        chunk_triples = graph_builder.extract_triples_for_chunks(chunks)
        graph = graph_builder.build_graph(chunks, chunk_triples)
        graph_builder.save_graph()
    else:
        graph = graph_builder.graph

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

    return questions, chunks, retriever, generator

def plot_subgraph(triples):
    if not triples:
        return None
    G = nx.DiGraph()
    for t in triples:
        G.add_edge(t["subject"], t["object"], label=t["relation"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#4C72B0", node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", width=1.5, arrowsize=15)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
    
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, font_color="#D62728")
    
    plt.axis("off")
    plt.tight_layout()
    return fig

# --- Main App ---
st.title("🕸️ GraphRAG vs Baseline RAG Explorer")
st.markdown("### Multi-Hop Question Answering via Vector Search + Knowledge Graph Traversal")

with st.spinner("Initializing GraphRAG pipeline and loading knowledge graph..."):
    questions, chunks, retriever, generator = load_system_pipeline()

# Sidebar Settings
st.sidebar.header("⚙️ Query Configuration")
preset_options = ["Custom Input"] + [f"Q{i+1}: {q['question']}" for i, q in enumerate(questions[:20])]
selected_preset = st.sidebar.selectbox("Select Sample Question Preset:", preset_options)

if selected_preset != "Custom Input":
    idx = int(selected_preset.split(":")[0].replace("Q", "")) - 1
    sample_q = questions[idx]
    default_q = sample_q["question"]
    gold_ans = sample_q["answer"]
    gold_titles = ", ".join(sample_q["gold_supporting_titles"])
else:
    default_q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    gold_ans = "N/A"
    gold_titles = "N/A"

user_question = st.sidebar.text_area("Enter Multi-Hop Question:", value=default_q, height=100)

top_k = st.sidebar.slider("Top K Chunks (k)", min_value=1, max_value=10, value=5)
traversal_hops = st.sidebar.slider("Graph Traversal Hops", min_value=1, max_value=3, value=2)

st.sidebar.markdown("---")
st.sidebar.markdown("**Gold Dataset Reference:**")
st.sidebar.write(f"**Gold Answer:** `{gold_ans}`")
st.sidebar.write(f"**Gold Paragraphs:** `{gold_titles}`")

if st.button("🚀 Run Comparison & Graph Traversal", type="primary", use_container_width=True):
    col1, col2 = st.columns(2)

    # 1. Baseline RAG Retrieval & Generation
    with col1:
        st.subheader("📌 Baseline Vector-Only RAG")
        with st.spinner("Running vector similarity search..."):
            b_chunks, _, b_debug = retriever.retrieve(user_question, top_k=top_k, mode="baseline")
            b_answer = generator.generate_answer(user_question, b_chunks, mode="baseline")

        st.success(f"**Baseline Answer:** {b_answer}")

        st.markdown("#### Retrieved Vector Chunks:")
        for idx, c in enumerate(b_chunks, 1):
            with st.expander(f"[{idx}] {c['title']}"):
                st.write(c['text'])

    # 2. Hybrid GraphRAG Retrieval & Generation
    with col2:
        st.subheader("🕸️ Hybrid GraphRAG")
        with st.spinner("Running question entity linking & graph traversal..."):
            h_chunks, h_triples, h_debug = retriever.retrieve(user_question, top_k=top_k, mode="hybrid", max_hops=traversal_hops)
            h_answer = generator.generate_answer(user_question, h_chunks, triples=h_triples, mode="hybrid")

        st.success(f"**Hybrid GraphRAG Answer:** {h_answer}")

        st.markdown(f"**Matched Entities:** `{', '.join(h_debug.get('matched_entities', []))}`")

        # Subgraph visual & triple table
        if h_triples:
            fig = plot_subgraph(h_triples)
            if fig:
                st.pyplot(fig)
            
            with st.expander(f"View Traversed Triples ({len(h_triples)} found)"):
                df_t = pd.DataFrame(h_triples)
                st.dataframe(df_t, use_container_width=True)

        st.markdown("#### Retrieved & Reranked Hybrid Chunks:")
        for idx, c in enumerate(h_chunks, 1):
            with st.expander(f"[{idx}] {c['title']}"):
                st.write(c['text'])

# Display Benchmarks Tab
st.markdown("---")
st.header("📊 Benchmark Evaluation Results")
if os.path.exists("results/evaluation_summary.csv"):
    df_eval = pd.read_csv("results/evaluation_summary.csv")
    st.dataframe(df_eval, use_container_width=True)
    if os.path.exists("results/metrics_comparison.png"):
        st.image("results/metrics_comparison.png", caption="System Metric Comparison (EM, F1, Recall@k)")
else:
    st.info("Run `python run_pipeline.py` to generate complete evaluation metrics & benchmark plots.")
