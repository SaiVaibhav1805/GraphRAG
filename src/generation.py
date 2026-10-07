from typing import List, Dict, Any
from src.llm_client import LLMClient

class RAGGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for idx, chunk in enumerate(chunks, 1):
            title = chunk.get("title", f"Document {idx}")
            text = chunk.get("text", "")
            formatted.append(f"[{idx}] Title: {title}\nContent: {text}")
        return "\n\n".join(formatted)

    def format_triples(self, triples: List[Dict[str, str]]) -> str:
        if not triples:
            return "No graph triples available."
        formatted = []
        for t in triples:
            formatted.append(f"({t['subject']}) --[{t['relation']}]--> ({t['object']})")
        return "\n".join(formatted)

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        triples: List[Dict[str, str]] = None,
        mode: str = "hybrid"
    ) -> str:
        context_str = self.format_chunks(chunks)

        if mode == "baseline" or not triples:
            prompt = f"""Answer the following multi-hop question accurately and concisely based strictly on the provided context paragraphs.
If the answer is a name, entity, date, or short phrase, output the answer directly.

Context Paragraphs:
{context_str}

Question: {question}
Answer:"""
        else:
            triples_str = self.format_triples(triples)
            prompt = f"""Answer the following multi-hop question accurately and concisely using both the retrieved context paragraphs and the Knowledge Graph entity relationships.

Knowledge Graph Triples:
{triples_str}

Context Paragraphs:
{context_str}

Question: {question}
Answer:"""

        system_instruction = "You are an expert multi-hop question answering system. Provide short, exact, factual answers based on provided documents and knowledge graph context."
        answer = self.llm.generate(prompt, system_instruction=system_instruction)
        return answer.strip()
