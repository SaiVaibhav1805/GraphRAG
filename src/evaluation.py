import os
import re
import json
import string
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple

def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        # If both are empty, EM/F1 is 1 if both empty, 0 otherwise
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)

def calculate_recall_at_k(retrieved_chunk_titles: List[str], gold_supporting_titles: List[str]) -> float:
    if not gold_supporting_titles:
        return 1.0
    retrieved_set = set([t.lower().strip() for t in retrieved_chunk_titles])
    gold_set = set([t.lower().strip() for t in gold_supporting_titles])
    hits = len(retrieved_set.intersection(gold_set))
    return float(hits / len(gold_set))


class Evaluator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_predictions(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes overall metrics and breakdowns for a list of evaluation records.
        Record keys: question_id, question, gold_answer, pred_answer, gold_titles, retrieved_titles, hop_count
        """
        em_list = []
        f1_list = []
        recall_list = []

        hop_breakdown = {}

        for rec in records:
            pred = rec["pred_answer"]
            gold = rec["gold_answer"]
            em = exact_match_score(pred, gold)
            f1 = f1_score(pred, gold)
            rec_k = calculate_recall_at_k(rec["retrieved_titles"], rec["gold_titles"])

            em_list.append(em)
            f1_list.append(f1)
            recall_list.append(rec_k)

            hop = rec.get("hop_count", 2)
            hop_key = "2-hop" if hop <= 2 else "3+ hop"
            if hop_key not in hop_breakdown:
                hop_breakdown[hop_key] = {"em": [], "f1": [], "recall": [], "count": 0}

            hop_breakdown[hop_key]["em"].append(em)
            hop_breakdown[hop_key]["f1"].append(f1)
            hop_breakdown[hop_key]["recall"].append(rec_k)
            hop_breakdown[hop_key]["count"] += 1

        summary = {
            "total_questions": len(records),
            "exact_match": float(np.mean(em_list)) if em_list else 0.0,
            "f1": float(np.mean(f1_list)) if f1_list else 0.0,
            "recall_at_k": float(np.mean(recall_list)) if recall_list else 0.0,
            "by_hop_count": {}
        }

        for hop_key, data in hop_breakdown.items():
            summary["by_hop_count"][hop_key] = {
                "count": data["count"],
                "exact_match": float(np.mean(data["em"])),
                "f1": float(np.mean(data["f1"])),
                "recall_at_k": float(np.mean(data["recall"]))
            }

        return summary

    def generate_plots_and_tables(self, comparison_results: Dict[str, Dict[str, Any]]):
        """
        Saves comparative metrics tables (CSV/JSON/MD) and plots (PNG) to results/.
        comparison_results: {"baseline": summary_dict, "graph_only": summary_dict, "hybrid": summary_dict}
        """
        # 1. Summary CSV Table
        rows = []
        for sys_name, summary in comparison_results.items():
            rows.append({
                "System": sys_name.upper(),
                "EM": round(summary["exact_match"] * 100, 2),
                "F1": round(summary["f1"] * 100, 2),
                "Recall@k": round(summary["recall_at_k"] * 100, 2),
                "2-hop F1": round(summary["by_hop_count"].get("2-hop", {}).get("f1", 0.0) * 100, 2),
                "3+ hop F1": round(summary["by_hop_count"].get("3+ hop", {}).get("f1", 0.0) * 100, 2)
            })

        df_res = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, "evaluation_summary.csv")
        json_path = os.path.join(self.results_dir, "evaluation_summary.json")
        md_path = os.path.join(self.results_dir, "results_table.md")

        df_res.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Performance Comparison Table\n\n")
            try:
                f.write(df_res.to_markdown(index=False))
            except Exception:
                f.write(df_res.to_string(index=False))
            f.write("\n")

        print(f"Saved evaluation tables to '{self.results_dir}'.")

        # 2. Main Metrics Bar Chart
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        metrics_df = df_res.melt(id_vars=["System"], value_vars=["EM", "F1", "Recall@k"], var_name="Metric", value_name="Score (%)")
        ax = sns.barplot(data=metrics_df, x="Metric", y="Score (%)", hue="System", palette="viridis")
        
        plt.title("GraphRAG vs Baseline Performance Metrics", fontsize=14, fontweight="bold", pad=15)
        plt.ylim(0, 105)
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                            textcoords='offset points')

        plot_path = os.path.join(self.results_dir, "metrics_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # 3. Hop Breakdown Bar Chart
        hop_rows = []
        for sys_name, summary in comparison_results.items():
            for hop_cat in ["2-hop", "3+ hop"]:
                if hop_cat in summary["by_hop_count"]:
                    hop_rows.append({
                        "System": sys_name.upper(),
                        "Hop Breakdown": hop_cat,
                        "F1 Score (%)": round(summary["by_hop_count"][hop_cat]["f1"] * 100, 2)
                    })
        
        if hop_rows:
            df_hop = pd.DataFrame(hop_rows)
            plt.figure(figsize=(8, 5))
            ax_h = sns.barplot(data=df_hop, x="Hop Breakdown", y="F1 Score (%)", hue="System", palette="magma")
            plt.title("F1 Score Improvement by Hop Count", fontsize=14, fontweight="bold", pad=15)
            plt.ylim(0, 105)
            
            for p in ax_h.patches:
                height = p.get_height()
                if height > 0:
                    ax_h.annotate(f"{height:.1f}%",
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                                textcoords='offset points')

            hop_plot_path = os.path.join(self.results_dir, "hop_count_breakdown.png")
            plt.tight_layout()
            plt.savefig(hop_plot_path, dpi=300)
            plt.close()

        print(f"Saved visualization plots to '{self.results_dir}'.")

if __name__ == "__main__":
    ev = Evaluator()
    dummy_records = [
        {"pred_answer": "American", "gold_answer": "American", "retrieved_titles": ["Scott Derrickson"], "gold_titles": ["Scott Derrickson"], "hop_count": 2},
        {"pred_answer": "Tim Burton", "gold_answer": "Tim Burton", "retrieved_titles": ["Ed Wood"], "gold_titles": ["Ed Wood"], "hop_count": 3}
    ]
    res = ev.evaluate_predictions(dummy_records)
    print("Test summary:", res)
