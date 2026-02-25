"""
Memory layer from WujiangXu/A-mem (https://github.com/WujiangXu/A-mem).
NeurIPS 2025: "A-Mem: Agentic Memory for LLM Agents".

Fixes applied vs upstream:
- Added missing `import re`
- Fixed undefined `e` reference in analyze_content inner except block
- Raised max_tokens 1000 → 4096 in OpenAIController to prevent truncated JSON
- Removed noisy debug print statements
"""

import re
import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import uuid

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from litellm import completion


def simple_tokenize(text: str) -> List[str]:
    return word_tokenize(text)


# ── LLM Controllers ───────────────────────────────────────────────────────────

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=4096,  # upstream hardcoded 1000, which truncates longer responses
        )
        return response.choices[0].message.content


class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        self.model = model

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type in ("number", "integer"):
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
        schema = response_format["json_schema"]["schema"]
        return {
            prop: self._generate_empty_value(prop_schema["type"], prop_schema.get("items"))
            for prop, prop_schema in schema.get("properties", {}).items()
        }

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            resp = completion(
                model=f"ollama_chat/{self.model}",
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
            return resp.choices[0].message.content
        except Exception:
            return json.dumps(self._generate_empty_response(response_format))


class LLMController:
    """Thin dispatcher: routes to the correct backend controller."""

    def __init__(
        self,
        backend: Literal["openai", "ollama"] = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("backend must be 'openai' or 'ollama'")


# ── Memory Note ───────────────────────────────────────────────────────────────

class MemoryNote:
    """Basic memory unit with LLM-generated metadata."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        links: Optional[List] = None,
        importance_score: Optional[float] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[List] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        llm_controller: Optional[LLMController] = None,
    ):
        self.content = content

        # Use LLM to fill in missing metadata fields
        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = MemoryNote.analyze_content(content, llm_controller)
            keywords = keywords or analysis["keywords"]
            context  = context  or analysis["context"]
            tags     = tags     or analysis["tags"]

        self.id               = id or str(uuid.uuid4())
        self.keywords         = keywords or []
        self.links            = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count  = retrieval_count or 0

        current_time      = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp    = timestamp     or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category          = category or "Uncategorized"
        self.tags              = tags     or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:
        """Call the LLM to extract keywords, context, and tags from content."""
        prompt = (
            "Generate a structured analysis of the following content by:\n"
            "1. Identifying the most salient keywords\n"
            "2. Extracting core themes and contextual elements\n"
            "3. Creating relevant categorical tags\n\n"
            'Format the response as a JSON object:\n'
            '{\n'
            '    "keywords": ["keyword1", "keyword2", ...],\n'
            '    "context": "one sentence summarising main topic/domain",\n'
            '    "tags": ["tag1", "tag2", ...]\n'
            '}\n\n'
            f"Content for analysis:\n{content}"
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "context":  {"type": "string"},
                        "tags":     {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["keywords", "context", "tags"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        try:
            raw = llm_controller.llm.get_completion(prompt, response_format=response_format)
            try:
                cleaned = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
                return json.loads(cleaned)
            except json.JSONDecodeError as parse_err:
                print(f"JSON parsing error in analyze_content: {parse_err}")
                return {"keywords": [], "context": "General", "tags": []}
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "category": "Uncategorized", "tags": []}


# ── Retriever ─────────────────────────────────────────────────────────────────

class SimpleEmbeddingRetriever:
    """Embedding-based retriever using cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model         = SentenceTransformer(model_name)
        self.corpus:       List[str]            = []
        self.embeddings:   Optional[np.ndarray] = None
        self.document_ids: Dict[str, int]       = {}

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return
        if not self.corpus:
            self.corpus     = list(documents)
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_emb = self.model.encode(documents)
            self.embeddings = (
                new_emb if self.embeddings is None
                else np.vstack([self.embeddings, new_emb])
            )
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> List[int]:
        """Return indices of the top-k most similar corpus entries."""
        if not self.corpus or self.embeddings is None:
            return []
        query_emb   = self.model.encode([query])[0]
        sims        = cosine_similarity([query_emb], self.embeddings)[0]
        k           = min(k, len(self.corpus))
        return np.argsort(sims)[-k:][::-1].tolist()

    @classmethod
    def load_from_local_memory(
        cls, memories: Dict, model_name: str
    ) -> "SimpleEmbeddingRetriever":
        all_docs = []
        for m in memories.values():
            meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            all_docs.append(f"{m.content} , {meta}")
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever


# ── Agentic Memory System ─────────────────────────────────────────────────────

_EVOLUTION_PROMPT = """\
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, along with its nearest neighbours.
Make decisions about its evolution.

New memory:
  context:  {context}
  content:  {content}
  keywords: {keywords}

Nearest-neighbour memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 strengthen: connect to related memories and update tags of this memory.
   2.2 update_neighbor: update context and tags of neighbour memories.
        If unchanged, repeat the original values.
        new_tags_neighborhood and new_context_neighborhood must each have exactly {neighbor_number} entries.

Return ONLY a JSON object:
{{
    "should_evolve": true or false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": [<integer indices>],
    "tags_to_update": ["tag_1", ..., "tag_n"],
    "new_context_neighborhood": ["new context", ...],
    "new_tags_neighborhood": [["tag_1", ...], ...]
}}
"""

_EVOLUTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "suggested_connections": {"type": "array", "items": {"type": "integer"}},
                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                "new_tags_neighborhood": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "required": [
                "should_evolve", "actions", "suggested_connections",
                "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval and LLM-driven evolution.

    Faithful port of WujiangXu/A-mem memory_layer.py with bug fixes.
    """

    def __init__(
        self,
        model_name:   str = "all-MiniLM-L6-v2",
        llm_backend:  str = "openai",
        llm_model:    str = "gpt-4o-mini",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        self.retriever      = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt        = 0
        self.evo_threshold  = evo_threshold
        self._model_name    = model_name

    # ── Public API ────────────────────────────────────────────────────────────

    def add_note(self, content: str, time: Optional[str] = None, **kwargs) -> str:
        """Add a new memory note; triggers evolution and optional consolidation."""
        note = MemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note

        doc = (
            f"content:{note.content} "
            f"context:{note.context} "
            f"keywords:{', '.join(note.keywords)} "
            f"tags:{', '.join(note.tags)}"
        )
        self.retriever.add_documents([doc])

        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        return note.id

    def consolidate_memories(self) -> None:
        """Rebuild the retriever index from all current memories."""
        self.retriever = SimpleEmbeddingRetriever(self._model_name)
        for memory in self.memories.values():
            meta = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([f"{memory.content} , {meta}"])

    def find_related_memories(self, query: str, k: int = 5):
        """Return (formatted_string, indices) for the k nearest memories."""
        if not self.memories:
            return "", []
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i < len(all_memories):
                m = all_memories[i]
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{m.timestamp}\t"
                    f"memory content: {m.content}\t"
                    f"memory context: {m.context}\t"
                    f"memory keywords: {m.keywords}\t"
                    f"memory tags: {m.tags}\n"
                )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Return a formatted string for retrieval (includes linked neighbours)."""
        if not self.memories:
            return ""
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                f"talk start time:{m.timestamp} "
                f"memory content: {m.content} "
                f"memory context: {m.context} "
                f"memory keywords: {m.keywords} "
                f"memory tags: {m.tags}\n"
            )
            j = 0
            for neighbor_idx in m.links:
                if isinstance(neighbor_idx, int) and neighbor_idx < len(all_memories) and j < k:
                    nb = all_memories[neighbor_idx]
                    memory_str += (
                        f"talk start time:{nb.timestamp} "
                        f"memory content: {nb.content} "
                        f"memory context: {nb.context} "
                        f"memory keywords: {nb.keywords} "
                        f"memory tags: {nb.tags}\n"
                    )
                    j += 1
        return memory_str

    # ── Evolution ─────────────────────────────────────────────────────────────

    def process_memory(self, note: MemoryNote):
        """Run the evolution LLM call; returns (should_evolve, updated_note)."""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        if not indices:
            return False, note

        prompt = _EVOLUTION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
        )
        raw = self.llm_controller.llm.get_completion(
            prompt, response_format=_EVOLUTION_RESPONSE_FORMAT
        )

        try:
            cleaned = raw.strip()
            if not cleaned.startswith("{"):
                start = cleaned.find("{")
                if start != -1:
                    cleaned = cleaned[start:]
            if not cleaned.endswith("}"):
                end = cleaned.rfind("}")
                if end != -1:
                    cleaned = cleaned[: end + 1]
            response_json = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Error in memory evolution: {e}")
            return False, note

        should_evolve = response_json.get("should_evolve", False)
        if should_evolve:
            all_notes  = list(self.memories.values())
            all_ids    = list(self.memories.keys())
            for action in response_json.get("actions", []):
                if action == "strengthen":
                    note.links.extend(response_json.get("suggested_connections", []))
                    note.tags = response_json.get("tags_to_update", note.tags)
                elif action == "update_neighbor":
                    new_contexts = response_json.get("new_context_neighborhood", [])
                    new_tags     = response_json.get("new_tags_neighborhood", [])
                    for i, idx in enumerate(indices):
                        if i >= len(new_tags):
                            break
                        if idx >= len(all_notes):
                            continue
                        nb = all_notes[idx]
                        nb.tags    = new_tags[i]
                        nb.context = new_contexts[i] if i < len(new_contexts) else nb.context
                        if idx < len(all_ids):
                            self.memories[all_ids[idx]] = nb

        return should_evolve, note
