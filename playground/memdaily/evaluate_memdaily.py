#!/usr/bin/env python3
"""
Evaluate memory systems on the MemDaily benchmark.

This script adapts the upstream MemSim/MemDaily JSON into the repo's existing
memory-evaluation style:
1. Store each MemDaily message into the target memory system.
2. Ask the graded question in a fresh conversation.
3. Save the retrieved context, model answer, and judge annotations.

It is designed to answer the main question for this repo:
"When a memory system fails on multi-hop daily-memory questions, is the failure
primarily retrieval, partial-chain coverage, distractor contamination, or final
reasoning?"

Examples:
    python playground/memdaily/evaluate_memdaily.py --system mem0 --limit 30
    python playground/memdaily/evaluate_memdaily.py --system simplemem --limit 30
    python playground/memdaily/evaluate_memdaily.py --system both --limit 60 \\
        --question-types conditional,post_processing,noisy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import requests
from dotenv import load_dotenv
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

try:
    import tiktoken
except ImportError:
    tiktoken = None

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent.parent


MEMDAILY_URL = (
    "https://raw.githubusercontent.com/nuster1128/MemSim/master/"
    "data_generation/final_dataset/memdaily.json"
)
DEFAULT_DATASET = PARENT_DIR / "datasets" / "long_hop" / "memdaily.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

QUESTION_TYPE_ALIASES = {
    "simple": "simple",
    "conditional": "conditional",
    "cond": "conditional",
    "comparative": "comparative",
    "comp": "comparative",
    "aggregative": "aggregative",
    "aggr": "aggregative",
    "post_processing": "post_processing",
    "post-processing": "post_processing",
    "post": "post_processing",
    "noisy": "noisy",
}
DEFAULT_QUESTION_TYPES = [
    "conditional",
    "comparative",
    "aggregative",
    "post_processing",
    "noisy",
]
MODEL_ALIASES = {
    "gpt-5.1-mini": "gpt-5.1",
}
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
INITIAL_RETRY_DELAY_SECONDS = 10
MAX_RETRY_DELAY_SECONDS = 300


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenized_length(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text.split())
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def extract_model_answer(response: str) -> str:
    if not response:
        return ""
    parts = re.split(r"\n\s*MEMORY_USED\s*:\s*", response, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def parse_csv_arg(raw: str) -> List[str]:
    parts = [part.strip() for part in raw.split(",")]
    return [part for part in parts if part]


def parse_support_set_plan(raw: str) -> Dict[int, int]:
    plan: Dict[int, int] = {}
    for part in parse_csv_arg(raw):
        size_text, sep, count_text = part.partition(":")
        if not sep:
            raise ValueError(
                "Invalid --support-set-plan entry "
                f"'{part}'. Expected SIZE:COUNT, for example 2:33,4:33,8:34."
            )
        support_set_size = int(size_text)
        count = int(count_text)
        if support_set_size <= 0 or count <= 0:
            raise ValueError("--support-set-plan sizes and counts must be positive integers.")
        if support_set_size in plan:
            raise ValueError(
                f"Duplicate support_set_size {support_set_size} in --support-set-plan."
            )
        plan[support_set_size] = count
    return plan


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        if "insufficient_quota" in str(exc).lower():
            return False
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def call_with_retry(label: str, fn):
    attempt = 1
    delay = INITIAL_RETRY_DELAY_SECONDS
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_retryable_error(exc):
                raise
            print(
                f"[retry] {label} failed on attempt {attempt}: {exc}. "
                f"Sleeping {delay}s before retry.",
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
            delay = min(int(delay * 1.5), MAX_RETRY_DELAY_SECONDS)


def resolve_question_types(raw: str) -> List[str]:
    resolved = []
    for name in parse_csv_arg(raw):
        key = name.lower()
        if key not in QUESTION_TYPE_ALIASES:
            raise ValueError(f"Unknown question type: {name}")
        resolved.append(QUESTION_TYPE_ALIASES[key])
    return resolved


def download_memdaily_dataset(dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(MEMDAILY_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest_path


@dataclass
class MemDailyExample:
    example_id: str
    question_type: str
    domain: str
    trajectory_id: int
    support_set_size: int
    contextual_statements: List[str]
    graded_question: str
    ground_truth_answer: str
    target_step_ids: List[int]
    target_messages: List[str]
    choices: Dict[str, str]


def load_memdaily_examples(
    dataset_path: Path,
    question_types: Iterable[str],
    domains: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    seed: int = 7,
    support_set_plan: Optional[Dict[int, int]] = None,
) -> List[MemDailyExample]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    domain_filter = set(domains) if domains else None
    examples: List[MemDailyExample] = []

    for question_type in question_types:
        if question_type not in raw:
            raise KeyError(f"Question type '{question_type}' not found in MemDaily JSON.")
        for domain, trajectories in raw[question_type].items():
            if domain_filter and domain not in domain_filter:
                continue
            for trajectory in trajectories:
                messages = trajectory["message_list"]
                qa = trajectory["QA"]
                target_ids = [int(v) for v in qa.get("target_step_id", [])]
                message_map = {int(item["mid"]): str(item["message"]) for item in messages}
                examples.append(
                    MemDailyExample(
                        example_id=f"memdaily-{question_type}-{domain}-{trajectory['tid']}",
                        question_type=question_type,
                        domain=domain,
                        trajectory_id=int(trajectory["tid"]),
                        support_set_size=len(target_ids),
                        contextual_statements=[str(item["message"]) for item in messages],
                        graded_question=str(qa["question"]),
                        ground_truth_answer=str(qa["answer"]),
                        target_step_ids=target_ids,
                        target_messages=[message_map[mid] for mid in target_ids if mid in message_map],
                        choices={str(k): str(v) for k, v in qa.get("choices", {}).items()},
                    )
                )

    if support_set_plan:
        selected_examples: List[MemDailyExample] = []
        selected_counts: Counter[int] = Counter()
        for example in examples:
            target_count = support_set_plan.get(example.support_set_size)
            if target_count is None:
                continue
            if selected_counts[example.support_set_size] >= target_count:
                continue
            selected_examples.append(example)
            selected_counts[example.support_set_size] += 1
            if all(selected_counts[size] >= count for size, count in support_set_plan.items()):
                break

        missing_counts = {
            size: count - selected_counts[size]
            for size, count in support_set_plan.items()
            if selected_counts[size] < count
        }
        if missing_counts:
            raise ValueError(
                "Not enough MemDaily examples matched --support-set-plan after filtering: "
                + ", ".join(
                    f"{size}->{missing} more needed" for size, missing in sorted(missing_counts.items())
                )
            )
        examples = selected_examples
    else:
        rng = random.Random(seed)
        rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


@dataclass
class JudgeResult:
    answer_correct: bool
    retrieval_status: str
    distractor_contamination: bool
    failure_mode: str
    reasoning: str


@dataclass
class ExampleTrace:
    example_id: str
    system: str
    question_type: str
    domain: str
    support_set_size: int
    graded_question: str
    ground_truth_answer: str
    target_step_ids: List[int]
    target_messages: List[str]
    all_memories_at_time: List[str]
    model_answer: str
    llm_response: str
    formatted_prompt: str
    retrieved_memories: str
    retrieval_time: float
    answer_time: float
    prompt_tokens: int
    retrieved_tokens: int
    judge_result: Dict[str, Any]


class Judge:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify(
        self,
        *,
        question_type: str,
        question: str,
        ground_truth_answer: str,
        target_messages: List[str],
        retrieved_memories: str,
        llm_response: str,
    ) -> JudgeResult:
        prompt = f"""You are grading failures on a memory benchmark.

Question type: {question_type}

Question:
{question}

Ground-truth answer:
{ground_truth_answer}

Target messages that should be retrieved/reasoned over:
{json.dumps(target_messages, ensure_ascii=False, indent=2)}

Retrieved memory context:
{retrieved_memories if retrieved_memories else "(none)"}

Model response:
{llm_response}

Return JSON with exactly these keys:
- answer_correct: boolean
- retrieval_status: one of ["full_chain_retrieved", "partial_chain_retrieved", "not_retrieved"]
- distractor_contamination: boolean
- failure_mode: one of [
  "correct_answer",
  "not_retrieved",
  "partial_chain_retrieved",
  "retrieved_full_chain_but_reasoning_failed",
  "distractor_contamination"
]
- reasoning: short explanation

Guidelines:
1. Mark answer_correct true if the model answer captures the same factual answer, even if phrased differently. The model answer may be shorter/more concise than the ground-truth answer but if it answers the question mark it as correct.
2. retrieval_status is full if the retrieved context contains enough information from all target messages to support the answer.
3. retrieval_status is partial if only some target messages or only part of the needed chain is present.
4. distractor_contamination is true when irrelevant retrieved context seems to have pushed the answer away from the correct one.
5. failure_mode must be "correct_answer" whenever answer_correct is true.
6. If answer is wrong and retrieval_status is full, use "retrieved_full_chain_but_reasoning_failed" unless distractor_contamination is clearly the primary issue.
7. Output JSON only."""

        response = call_with_retry(
            label=f"judge classify for {question_type}",
            fn=lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            ),
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        return JudgeResult(
            answer_correct=bool(parsed.get("answer_correct", False)),
            retrieval_status=str(parsed.get("retrieval_status", "not_retrieved")),
            distractor_contamination=bool(parsed.get("distractor_contamination", False)),
            failure_mode=str(parsed.get("failure_mode", "not_retrieved")),
            reasoning=str(parsed.get("reasoning", "")),
        )


class BaseHarness:
    name = "base"

    def __init__(self, llm_model: str, api_key: str, retrieval_limit: int, work_dir: Path):
        self.answer_client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.retrieval_limit = retrieval_limit
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def store_messages(self, messages: List[str], example_id: str) -> None:
        raise NotImplementedError

    def retrieve_memories(self, question: str) -> str:
        raise NotImplementedError

    def get_all_memories(self) -> List[str]:
        raise NotImplementedError

    def teardown(self) -> None:
        pass

    def answer(self, question: str, retrieved_memories: str) -> tuple[str, str]:
        prompt = f"""You are answering a personal-memory question from retrieved memory snippets.

Question:
{question}

Retrieved memories:
{retrieved_memories if retrieved_memories else "(none)"}

Instructions:
- Answer concisely in the same language as the question if possible.
- Use the retrieved memories when they are sufficient.
- If the retrieved memories are insufficient, say that the answer is not available from the provided memories.
- After the answer, add a final line that starts with exactly "MEMORY_USED:" and briefly quote or paraphrase the memories you relied on, or write "none".
"""
        response = call_with_retry(
            label="answer generation",
            fn=lambda: self.answer_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        return prompt, response.choices[0].message.content or ""


class Mem0Harness(BaseHarness):
    name = "mem0"

    def __init__(
        self,
        llm_model: str,
        api_key: str,
        retrieval_limit: int,
        work_dir: Path,
        mem0_llm_provider: str = "openai",
        mem0_llm_model: str = "gpt-5.1",
        mem0_embedding_provider: Optional[str] = None,
        mem0_embedding_model: Optional[str] = None,
        mem0_ollama_base_url: Optional[str] = None,
    ):
        super().__init__(llm_model=llm_model, api_key=api_key, retrieval_limit=retrieval_limit, work_dir=work_dir)
        uid = uuid4().hex
        self.mem0_dir = self.work_dir / f"mem0_home_{uid}"
        self.mem0_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MEM0_DIR"] = str(self.mem0_dir)
        os.environ["MEM0_TELEMETRY"] = "False"
        import mem0
        from mem0.configs.base import MemoryConfig
        from mem0.embeddings.configs import EmbedderConfig
        from mem0.llms.configs import LlmConfig

        self.user_id = f"memdaily_{uid}"
        self.qdrant_path = self.work_dir / f"qdrant_{uid}"
        self.history_db_path = self.work_dir / f"history_{uid}.db"

        llm_cfg = LlmConfig(provider=mem0_llm_provider, config={"model": mem0_llm_model})
        config_kwargs: Dict[str, Any] = {"llm": llm_cfg}

        if mem0_embedding_provider and mem0_embedding_model:
            embedder_cfg_data: Dict[str, Any] = {"model": mem0_embedding_model}
            if mem0_embedding_provider == "ollama" and mem0_ollama_base_url:
                embedder_cfg_data["base_url"] = mem0_ollama_base_url
            config_kwargs["embedder"] = EmbedderConfig(
                provider=mem0_embedding_provider,
                config=embedder_cfg_data,
            )

        config = MemoryConfig(**config_kwargs)
        config.vector_store.config.path = str(self.qdrant_path)
        config.history_db_path = str(self.history_db_path)
        self.memory = mem0.Memory(config=config)

    def store_messages(self, messages: List[str], example_id: str) -> None:
        for idx, message in enumerate(messages, 1):
            call_with_retry(
                label=f"mem0 store {example_id} message {idx}/{len(messages)}",
                fn=lambda message=message: self.memory.add(
                    [{"role": "user", "content": message}],
                    user_id=self.user_id,
                ),
            )

    def retrieve_memories(self, question: str) -> str:
        results = call_with_retry(
            label="mem0 retrieve memories",
            fn=lambda: self.memory.search(
                query=question,
                user_id=self.user_id,
                limit=self.retrieval_limit,
            ),
        )
        memories = results.get("results", [])
        return "\n".join(f"- {item['memory']}" for item in memories)

    def get_all_memories(self) -> List[str]:
        result = call_with_retry(
            label="mem0 get all memories",
            fn=lambda: self.memory.get_all(user_id=self.user_id, limit=10000),
        )
        return [entry["memory"] for entry in result.get("results", [])]

    def teardown(self) -> None:
        try:
            self.memory.delete_all(user_id=self.user_id)
        except Exception:
            pass
        try:
            shutil.rmtree(self.qdrant_path, ignore_errors=True)
        except Exception:
            pass
        try:
            self.history_db_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(self.mem0_dir, ignore_errors=True)
        except Exception:
            pass


class SimpleMemHarness(BaseHarness):
    name = "simplemem"

    def __init__(self, llm_model: str, api_key: str, retrieval_limit: int, work_dir: Path):
        super().__init__(llm_model=llm_model, api_key=api_key, retrieval_limit=retrieval_limit, work_dir=work_dir)
        import sys

        simplemem_dir = PARENT_DIR / "SimpleMem"
        if not simplemem_dir.exists():
            raise FileNotFoundError(
                "SimpleMem directory not found. Add the SimpleMem submodule/repo to "
                f"{simplemem_dir} before running --system simplemem."
            )
        if str(simplemem_dir) not in sys.path:
            sys.path.insert(0, str(simplemem_dir))

        from main import SimpleMemSystem  # type: ignore

        self.db_path = self.work_dir / f"simplemem_{uuid4().hex}"
        self.memory_system = SimpleMemSystem(
            api_key=api_key,
            model=llm_model,
            db_path=str(self.db_path),
            clear_db=True,
        )

    def store_messages(self, messages: List[str], example_id: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for message in messages:
            self.memory_system.add_dialogue("User", message, timestamp)
        self.memory_system.finalize()

    def retrieve_memories(self, question: str) -> str:
        contexts = self.memory_system.hybrid_retriever.retrieve(question)
        if not contexts:
            return ""
        return self.memory_system.answer_generator._format_contexts(contexts)

    def get_all_memories(self) -> List[str]:
        parts: List[str] = []
        for entry in self.memory_system.get_all_memories():
            line = entry.lossless_restatement
            extras = []
            if getattr(entry, "topic", None):
                extras.append(f"topic: {entry.topic}")
            if getattr(entry, "timestamp", None):
                extras.append(f"time: {entry.timestamp}")
            if extras:
                line = f"{line} ({'; '.join(extras)})"
            parts.append(line)
        return parts

    def teardown(self) -> None:
        try:
            shutil.rmtree(self.db_path, ignore_errors=True)
        except Exception:
            pass


def build_harness(
    system: str,
    llm_model: str,
    api_key: str,
    retrieval_limit: int,
    work_dir: Path,
    mem0_llm_provider: str = "openai",
    mem0_llm_model: str = "gpt-5.1",
    mem0_embedding_provider: Optional[str] = None,
    mem0_embedding_model: Optional[str] = None,
    mem0_ollama_base_url: Optional[str] = None,
) -> BaseHarness:
    if system == "mem0":
        return Mem0Harness(
            llm_model=llm_model,
            api_key=api_key,
            retrieval_limit=retrieval_limit,
            work_dir=work_dir,
            mem0_llm_provider=mem0_llm_provider,
            mem0_llm_model=mem0_llm_model,
            mem0_embedding_provider=mem0_embedding_provider,
            mem0_embedding_model=mem0_embedding_model,
            mem0_ollama_base_url=mem0_ollama_base_url,
        )
    if system == "simplemem":
        return SimpleMemHarness(llm_model=llm_model, api_key=api_key, retrieval_limit=retrieval_limit, work_dir=work_dir)
    raise ValueError(f"Unsupported system: {system}")


def aggregate_traces(traces: List[ExampleTrace]) -> Dict[str, Any]:
    def summarize(bucket: List[ExampleTrace]) -> Dict[str, Any]:
        if not bucket:
            return {}
        failure_counts = Counter(trace.judge_result["failure_mode"] for trace in bucket)
        retrieval_counts = Counter(trace.judge_result["retrieval_status"] for trace in bucket)
        answer_correct = sum(bool(trace.judge_result["answer_correct"]) for trace in bucket)
        distractor_count = sum(bool(trace.judge_result["distractor_contamination"]) for trace in bucket)
        return {
            "count": len(bucket),
            "answer_accuracy": answer_correct / len(bucket),
            "full_chain_retrieval_rate": retrieval_counts["full_chain_retrieved"] / len(bucket),
            "partial_chain_retrieval_rate": retrieval_counts["partial_chain_retrieved"] / len(bucket),
            "not_retrieved_rate": retrieval_counts["not_retrieved"] / len(bucket),
            "distractor_contamination_rate": distractor_count / len(bucket),
            "avg_retrieval_time": sum(trace.retrieval_time for trace in bucket) / len(bucket),
            "avg_answer_time": sum(trace.answer_time for trace in bucket) / len(bucket),
            "avg_prompt_tokens": sum(trace.prompt_tokens for trace in bucket) / len(bucket),
            "avg_retrieved_tokens": sum(trace.retrieved_tokens for trace in bucket) / len(bucket),
            "failure_mode_counts": dict(failure_counts),
            "retrieval_status_counts": dict(retrieval_counts),
        }

    by_question_type: Dict[str, List[ExampleTrace]] = defaultdict(list)
    by_domain: Dict[str, List[ExampleTrace]] = defaultdict(list)
    by_support_set_size: Dict[int, List[ExampleTrace]] = defaultdict(list)
    for trace in traces:
        by_question_type[trace.question_type].append(trace)
        by_domain[trace.domain].append(trace)
        by_support_set_size[trace.support_set_size].append(trace)

    return {
        "overall": summarize(traces),
        "by_question_type": {k: summarize(v) for k, v in sorted(by_question_type.items())},
        "by_domain": {k: summarize(v) for k, v in sorted(by_domain.items())},
        "by_support_set_size": {
            str(k): summarize(v) for k, v in sorted(by_support_set_size.items())
        },
    }


def build_example_reviews(traces: List[ExampleTrace]) -> List[Dict[str, Any]]:
    reviews = []
    for trace in traces:
        reviews.append(
            {
                "example_id": trace.example_id,
                "question_type": trace.question_type,
                "domain": trace.domain,
                "support_set_size": trace.support_set_size,
                "question": trace.graded_question,
                "correct_answer": trace.ground_truth_answer,
                "model_answer": trace.model_answer,
                "raw_model_response": trace.llm_response,
                "failure_mode": trace.judge_result["failure_mode"],
                "retrieval_status": trace.judge_result["retrieval_status"],
                "answer_correct": trace.judge_result["answer_correct"],
            }
        )
    return reviews


def run_system(
    *,
    system: str,
    examples: List[MemDailyExample],
    api_key: str,
    llm_model: str,
    judge_model: str,
    retrieval_limit: int,
    output_dir: Path,
    mem0_llm_provider: str = "openai",
    mem0_llm_model: str = "gpt-5.1",
    mem0_embedding_provider: Optional[str] = None,
    mem0_embedding_model: Optional[str] = None,
    mem0_ollama_base_url: Optional[str] = None,
    support_set_plan: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    judge = Judge(api_key=api_key, model=judge_model)
    traces: List[ExampleTrace] = []
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, example in enumerate(examples, 1):
        print(
            f"[{system}] {idx}/{len(examples)} "
            f"{example.question_type}/{example.domain} "
            f"{example.example_id}",
            flush=True,
        )
        harness = build_harness(
            system=system,
            llm_model=llm_model,
            api_key=api_key,
            retrieval_limit=retrieval_limit,
            work_dir=tmp_dir,
            mem0_llm_provider=mem0_llm_provider,
            mem0_llm_model=mem0_llm_model,
            mem0_embedding_provider=mem0_embedding_provider,
            mem0_embedding_model=mem0_embedding_model,
            mem0_ollama_base_url=mem0_ollama_base_url,
        )
        try:
            harness.store_messages(example.contextual_statements, example.example_id)
            all_memories_at_time = harness.get_all_memories()

            retrieval_start = time.time()
            retrieved_memories = harness.retrieve_memories(example.graded_question)
            retrieval_time = time.time() - retrieval_start

            answer_start = time.time()
            formatted_prompt, llm_response = harness.answer(example.graded_question, retrieved_memories)
            answer_time = time.time() - answer_start
            model_answer = extract_model_answer(llm_response)

            judge_result = judge.classify(
                question_type=example.question_type,
                question=example.graded_question,
                ground_truth_answer=example.ground_truth_answer,
                target_messages=example.target_messages,
                retrieved_memories=retrieved_memories,
                llm_response=model_answer,
            )

            traces.append(
                ExampleTrace(
                    example_id=example.example_id,
                    system=system,
                    question_type=example.question_type,
                    domain=example.domain,
                    support_set_size=example.support_set_size,
                    graded_question=example.graded_question,
                    ground_truth_answer=example.ground_truth_answer,
                    target_step_ids=example.target_step_ids,
                    target_messages=example.target_messages,
                    all_memories_at_time=all_memories_at_time,
                    model_answer=model_answer,
                    llm_response=llm_response,
                    formatted_prompt=formatted_prompt,
                    retrieved_memories=retrieved_memories,
                    retrieval_time=retrieval_time,
                    answer_time=answer_time,
                    prompt_tokens=tokenized_length(formatted_prompt),
                    retrieved_tokens=tokenized_length(retrieved_memories),
                    judge_result=asdict(judge_result),
                )
            )
        finally:
            harness.teardown()

    summary = {
        "system": system,
        "llm_model": llm_model,
        "judge_model": judge_model,
        "num_examples": len(examples),
        "aggregates": aggregate_traces(traces),
        "example_reviews": build_example_reviews(traces),
    }
    return {
        "summary": summary,
        "traces": [asdict(trace) for trace in traces],
        "run_metadata": {
            "system": system,
            "llm_model": llm_model,
            "judge_model": judge_model,
            "num_examples": len(examples),
            "support_set_plan": support_set_plan or {},
            "mem0_llm_provider": mem0_llm_provider,
            "mem0_llm_model": mem0_llm_model,
            "mem0_embedding_provider": mem0_embedding_provider,
            "mem0_embedding_model": mem0_embedding_model,
            "timestamp": datetime.now().isoformat(),
        },
    }


def save_results(result: Dict[str, Any], output_dir: Path, system: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"{system}_summary_{timestamp}.json"
    traces_path = output_dir / f"{system}_traces_{timestamp}.json"
    csv_path = output_dir / f"{system}_classifications_{timestamp}.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": result.get("run_metadata", {}),
                "summary": result["summary"],
                "graded_traces": result["traces"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "question_type",
                "domain",
                "support_set_size",
                "question",
                "correct_answer",
                "model_answer",
                "answer_correct",
                "retrieval_status",
                "failure_mode",
                "distractor_contamination",
            ]
        )
        for trace in result["traces"]:
            judge_result = trace["judge_result"]
            writer.writerow(
                [
                    trace["example_id"],
                    trace["question_type"],
                    trace["domain"],
                    trace["support_set_size"],
                    trace["graded_question"],
                    trace["ground_truth_answer"],
                    trace["model_answer"],
                    judge_result["answer_correct"],
                    judge_result["retrieval_status"],
                    judge_result["failure_mode"],
                    judge_result["distractor_contamination"],
                ]
            )

    print(f"Saved summary: {summary_path}")
    print(f"Saved traces: {traces_path}")
    print(f"Saved classifications: {csv_path}")


def print_summary(summary: Dict[str, Any]) -> None:
    overall = summary["aggregates"]["overall"]
    print("=" * 80)
    print(f"MEMDAILY SUMMARY - {summary['system']}")
    print("=" * 80)
    print(f"Examples: {summary['num_examples']}")
    print(f"Answer accuracy: {overall['answer_accuracy']:.1%}")
    print(f"Full-chain retrieval: {overall['full_chain_retrieval_rate']:.1%}")
    print(f"Partial-chain retrieval: {overall['partial_chain_retrieval_rate']:.1%}")
    print(f"Not retrieved: {overall['not_retrieved_rate']:.1%}")
    print(f"Distractor contamination: {overall['distractor_contamination_rate']:.1%}")
    print("\nFailure modes:")
    for key, value in sorted(overall["failure_mode_counts"].items()):
        print(f"  {key}: {value}")
    print("\nBy question type:")
    for question_type, stats in summary["aggregates"]["by_question_type"].items():
        print(
            f"  {question_type:16s} "
            f"acc={stats['answer_accuracy']:.1%} "
            f"full={stats['full_chain_retrieval_rate']:.1%} "
            f"partial={stats['partial_chain_retrieval_rate']:.1%} "
            f"not_ret={stats['not_retrieved_rate']:.1%}"
        )
    print("\nBy support_set_size:")
    for support_set_size, stats in summary["aggregates"]["by_support_set_size"].items():
        print(
            f"  {support_set_size:16s} "
            f"acc={stats['answer_accuracy']:.1%} "
            f"full={stats['full_chain_retrieval_rate']:.1%} "
            f"partial={stats['partial_chain_retrieval_rate']:.1%} "
            f"not_ret={stats['not_retrieved_rate']:.1%}"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory systems on MemDaily.")
    parser.add_argument(
        "--system",
        choices=["mem0", "simplemem", "both"],
        default="mem0",
        help="Memory system to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to memdaily.json. Downloads automatically if missing unless disabled.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the upstream MemDaily JSON if the dataset path does not exist.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default=",".join(DEFAULT_QUESTION_TYPES),
        help="Comma-separated question types to evaluate.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Optional comma-separated MemDaily domains (roles, events, items, places, hybrid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of examples to evaluate after filtering/shuffling.",
    )
    parser.add_argument(
        "--support-set-plan",
        type=str,
        default="",
        help=(
            "Optional exact support_set_size selection plan in dataset order, formatted as "
            "SIZE:COUNT pairs, for example 2:33,4:33,8:34."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.1-mini",
        help="Model used to answer benchmark questions.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.1-mini",
        help="Model used to classify failures.",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=5,
        help="Top-k memories to retrieve from the memory system.",
    )
    parser.add_argument(
        "--mem0-llm-provider",
        type=str,
        default="openai",
        help="LLM provider for mem0 memory operations.",
    )
    parser.add_argument(
        "--mem0-llm-model",
        type=str,
        default="gpt-5.1-mini",
        help="LLM model for mem0 memory operations.",
    )
    parser.add_argument(
        "--mem0-embedding-provider",
        type=str,
        default=None,
        help="Embedding provider for mem0 memory operations.",
    )
    parser.add_argument(
        "--mem0-embedding-model",
        type=str,
        default=None,
        help="Embedding model for mem0 memory operations.",
    )
    parser.add_argument(
        "--mem0-ollama-base-url",
        type=str,
        default=None,
        help="Ollama base URL when using --mem0-embedding-provider=ollama.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summaries and traces.",
    )
    args = parser.parse_args()

    load_dotenv(PARENT_DIR / ".env")
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required via OPENAI_KEY or OPENAI_API_KEY.")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PARENT_DIR / dataset_path
    if not dataset_path.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. "
                "Pass --download-if-missing to fetch the upstream MemDaily JSON."
            )
        print(f"Downloading MemDaily to {dataset_path}...")
        download_memdaily_dataset(dataset_path)

    question_types = resolve_question_types(args.question_types)
    domains = parse_csv_arg(args.domains) if args.domains else None
    support_set_plan = parse_support_set_plan(args.support_set_plan)
    examples = load_memdaily_examples(
        dataset_path=dataset_path,
        question_types=question_types,
        domains=domains,
        limit=args.limit,
        seed=args.seed,
        support_set_plan=support_set_plan or None,
    )
    if not examples:
        raise ValueError("No MemDaily examples matched the provided filters.")
    if support_set_plan:
        selected_counts = Counter(example.support_set_size for example in examples)
        print(
            "Selected support_set_size counts: "
            + ", ".join(
                f"{size}={selected_counts[size]}/{count}"
                for size, count in sorted(support_set_plan.items())
            )
        )

    resolved_llm_model = resolve_model_name(args.llm_model)
    resolved_judge_model = resolve_model_name(args.judge_model)
    resolved_mem0_llm_model = resolve_model_name(args.mem0_llm_model)
    if resolved_llm_model != args.llm_model:
        print(f"Resolved llm model alias: {args.llm_model} -> {resolved_llm_model}")
    if resolved_judge_model != args.judge_model:
        print(f"Resolved judge model alias: {args.judge_model} -> {resolved_judge_model}")
    if resolved_mem0_llm_model != args.mem0_llm_model:
        print(f"Resolved mem0 llm model alias: {args.mem0_llm_model} -> {resolved_mem0_llm_model}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PARENT_DIR / output_dir

    systems = ["mem0", "simplemem"] if args.system == "both" else [args.system]
    for system in systems:
        result = run_system(
            system=system,
            examples=examples,
            api_key=api_key,
            llm_model=resolved_llm_model,
            judge_model=resolved_judge_model,
            retrieval_limit=args.num_memories,
            output_dir=output_dir,
            mem0_llm_provider=args.mem0_llm_provider,
            mem0_llm_model=resolved_mem0_llm_model,
            mem0_embedding_provider=args.mem0_embedding_provider,
            mem0_embedding_model=args.mem0_embedding_model,
            mem0_ollama_base_url=args.mem0_ollama_base_url,
            support_set_plan=support_set_plan or None,
        )
        print_summary(result["summary"])
        save_results(result, output_dir=output_dir, system=system)


if __name__ == "__main__":
    main()
