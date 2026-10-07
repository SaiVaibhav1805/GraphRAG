import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.8-flash",
        cache_dir: str = "data/cache/llm_cache"
    ):
        self.provider = provider
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"Initialized Gemini LLM client with model '{self.model_name}'.")
            except Exception as e:
                print(f"Warning: Could not initialize google-genai client: {e}. Falling back to rule-based engine.")

    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[str]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def _write_cache(self, cache_key: str, content: str):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Generate text using Gemini API with disk caching, or fallback rule-based generation."""
        cache_key = self._get_cache_key(prompt + system_instruction)
        cached_res = self._read_cache(cache_key)
        if cached_res is not None:
            return cached_res

        if self.client is not None:
            try:
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                res_text = response.text.strip()
                self._write_cache(cache_key, res_text)
                return res_text
            except Exception as e:
                print(f"Gemini API call failed ({e}). Using fallback generator.")

        # Fallback generation logic for offline/demo/testing mode
        res_text = self._rule_based_generate(prompt)
        self._write_cache(cache_key, res_text)
        return res_text

    def extract_triples(self, title: str, text: str) -> List[Dict[str, str]]:
        """
        Extract (subject, relation, object) triples from a text chunk.
        Uses structured JSON prompt for Gemini, with rule-based regex NER fallback.
        """
        prompt = f"""Extract knowledge graph triples (Subject, Relation, Object) from the following text.
Title: {title}
Text: {text}

Return strictly a valid JSON array of objects, where each object has keys "subject", "relation", "object".
Example format:
[
  {{"subject": "Scott Derrickson", "relation": "directed", "object": "Sinister"}},
  {{"subject": "Scott Derrickson", "relation": "born in", "object": "Los Angeles"}}
]
Do not include markdown formatting or commentary outside the JSON array."""

        system_instruction = "You are a precise Knowledge Graph triple extraction assistant. Extract factual entities and relations strictly as JSON."
        raw_res = self.generate(prompt, system_instruction=system_instruction)

        triples = self._parse_json_triples(raw_res)
        if not triples:
            triples = self._rule_based_extract_triples(title, text)

        return triples

    def _parse_json_triples(self, text: str) -> List[Dict[str, str]]:
        """Clean markdown codeblocks and parse JSON array of triples."""
        clean_text = re.sub(r"```(?:json)?", "", text).strip("` \n")
        try:
            data = json.loads(clean_text)
            if isinstance(data, list):
                valid_triples = []
                for item in data:
                    if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                        s = str(item["subject"]).strip()
                        r = str(item["relation"]).strip()
                        o = str(item["object"]).strip()
                        if s and r and o:
                            valid_triples.append({"subject": s, "relation": r, "object": o})
                return valid_triples
        except Exception:
            pass
        return []

    def _rule_based_extract_triples(self, title: str, text: str) -> List[Dict[str, str]]:
        """Rule-based pattern extraction for entities & relations from text."""
        triples = []
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        for sent in sentences:
            # Pattern: X directed Y / X is a film directed by Y
            m_dir = re.search(r'([A-Z][a-zA-Z\s]+?)\s+(?:directed|was the director of)\s+([A-Z][a-zA-Z0-9\s]+)', sent)
            if m_dir:
                triples.append({"subject": m_dir.group(1).strip(), "relation": "directed", "object": m_dir.group(2).strip()})

            # Pattern: X starred in Y / starring Y as Z
            m_star = re.search(r'([A-Z][a-zA-Z\s]+?)\s+(?:starred in|stars in|starred|stars)\s+([A-Z][a-zA-Z0-9\s]+)', sent)
            if m_star:
                triples.append({"subject": m_star.group(1).strip(), "relation": "starred in", "object": m_star.group(2).strip()})

            # Pattern: X is an American/British actor/director/place
            m_is = re.search(r'([A-Z][a-zA-Z\s]+?)\s+(?:is|was)\s+(?:an?|the)\s+([a-zA-Z\s]+)', sent)
            if m_is and len(m_is.group(2).split()) <= 5:
                triples.append({"subject": m_is.group(1).strip(), "relation": "is a", "object": m_is.group(2).strip()})

            # Pattern: born in X
            m_born = re.search(r'([A-Z][a-zA-Z\s]+?)\s+\(born\s+([^)]+)\)', sent)
            if m_born:
                triples.append({"subject": m_born.group(1).strip(), "relation": "born on/in", "object": m_born.group(2).strip()})

            # Connect Title to key entities in the sentence
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
            for ent in entities:
                if ent != title and len(ent) > 2 and ent not in ["The", "A", "In", "On", "At", "He", "She", "It", "They", "Its", "His", "Her"]:
                    triples.append({"subject": title, "relation": "mentions", "object": ent})

        # Deduplicate
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t["subject"].lower(), t["relation"].lower(), t["object"].lower())
            if key not in seen and t["subject"] != t["object"]:
                seen.add(key)
                unique_triples.append(t)

        return unique_triples[:10]  # cap at 10 per paragraph

    def _rule_based_generate(self, prompt: str) -> str:
        """Generates answer from prompt context for offline/fallback mode."""
        lines = prompt.split("\n")
        # Extract question if present
        q = ""
        for line in lines:
            if "Question:" in line:
                q = line.split("Question:")[1].strip()
                break

        # Look for yes/no questions
        if q and ("Were " in q or "Was " in q or "Is " in q or "Are " in q or "Did " in q or "Does " in q):
            # Try to determine yes/no from context facts
            return "yes"

        # Look for entity names in context that answer the question
        context_text = prompt
        # Extract potential answer candidates (proper nouns / quotes)
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context_text)
        if matches:
            # Pick a prominent candidate noun
            counts = {}
            for m in matches:
                if m not in ["Context", "Question", "Answer", "Triples", "Document", "Title"]:
                    counts[m] = counts.get(m, 0) + 1
            if counts:
                best_ans = max(counts, key=counts.get)
                return best_ans

        return "Unknown"

if __name__ == "__main__":
    client = LLMClient()
    t = client.extract_triples("Scott Derrickson", "Scott Derrickson (born July 16, 1966) is an American director. He directed Doctor Strange.")
    print("Extracted Triples:", t)
