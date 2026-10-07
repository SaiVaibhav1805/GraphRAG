import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

class DataIngestion:
    def __init__(self, parquet_path: str, sample_size: int = 500, random_seed: int = 42, processed_dir: str = "data/processed"):
        self.parquet_path = parquet_path
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found at {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        return df

    def sample_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample a fixed subset of questions using a deterministic seed."""
        if len(df) <= self.sample_size:
            return df
        sampled_df = df.sample(n=self.sample_size, random_state=self.random_seed).copy()
        sampled_df.reset_index(drop=True, inplace=True)
        return sampled_df

    def process_and_pool_corpus(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts questions and pools all unique paragraphs into a shared corpus.
        Returns:
            questions: List of question dicts with gold answers & supporting facts
            chunks: List of unique paragraph chunk dicts
        """
        questions = []
        chunk_map = {}  # key: title, value: chunk_dict

        for idx, row in df.iterrows():
            q_id = str(row['id'])
            question_text = str(row['question'])
            answer_text = str(row['answer'])
            q_type = str(row.get('type', 'unknown'))
            level = str(row.get('level', 'unknown'))

            # Parse supporting facts
            sup_facts_raw = row['supporting_facts']
            sup_titles = list(sup_facts_raw.get('title', []))
            sup_sent_ids = list(sup_facts_raw.get('sent_id', []))
            supporting_facts = list(zip(sup_titles, [int(x) for x in sup_sent_ids]))
            gold_supporting_titles = list(set(sup_titles))

            # Infer hop count
            # Comparison questions are usually 2-hop; bridge questions depend on supporting titles count
            hop_count = max(2, len(gold_supporting_titles))

            # Process context paragraphs
            ctx_raw = row['context']
            ctx_titles = list(ctx_raw.get('title', []))
            ctx_sentences = list(ctx_raw.get('sentences', []))

            q_chunk_ids = []
            for title, sents in zip(ctx_titles, ctx_sentences):
                clean_title = str(title).strip()
                sent_list = [str(s).strip() for s in sents if str(s).strip()]
                full_text = " ".join(sent_list)
                
                # Unique ID per paragraph title
                chunk_id = f"doc_{hash(clean_title) & 0xffffffff:08x}"

                if clean_title not in chunk_map:
                    chunk_map[clean_title] = {
                        "chunk_id": chunk_id,
                        "title": clean_title,
                        "text": full_text,
                        "sentences": sent_list
                    }
                q_chunk_ids.append(chunk_map[clean_title]["chunk_id"])

            questions.append({
                "question_id": q_id,
                "question": question_text,
                "answer": answer_text,
                "type": q_type,
                "level": level,
                "hop_count": hop_count,
                "supporting_facts": supporting_facts,
                "gold_supporting_titles": gold_supporting_titles,
                "context_chunk_ids": q_chunk_ids
            })

        chunks = list(chunk_map.values())
        return questions, chunks

    def run(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs end-to-end data ingestion and saves processed data."""
        df_raw = self.load_raw_dataset()
        df_sampled = self.sample_dataset(df_raw)
        questions, chunks = self.process_and_pool_corpus(df_sampled)

        # Save to disk
        q_path = os.path.join(self.processed_dir, "questions.json")
        c_path = os.path.join(self.processed_dir, "chunks.json")

        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"Data ingestion complete: {len(questions)} questions, {len(chunks)} unique corpus chunks saved to '{self.processed_dir}'.")
        return questions, chunks

if __name__ == "__main__":
    ingestion = DataIngestion(parquet_path="validation-00000-of-00001.parquet", sample_size=500)
    ingestion.run()
