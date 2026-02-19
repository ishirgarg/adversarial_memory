"""
Evaluation framework for memory systems.
"""

import time
from dataclasses import dataclass, field
from typing import List

from .chat import ChatSystem
from .dataset import ChatDataset, ConversationData
from .tokenizer import EvaluationTokenizer
from .types import (
    ConversationID,
    EvaluationPromptTemplate,
    LLM,
    LLMResponse,
    MemorySystem,
    Prompt,
)


@dataclass
class QueryTrace:
    """Full trace of a single query turn within a conversation."""

    # The original user query
    query: Prompt
    # Whether this turn was flagged for grading
    should_grade: bool
    # Raw memories/context string returned by the memory system
    retrieved_memories: str
    # The fully-formatted prompt sent to the LLM
    formatted_prompt: str
    # The LLM's response
    response: LLMResponse
    # Token counts for this turn
    input_tokens: int
    output_tokens: int
    # Timing breakdown for this turn (seconds)
    retrieval_time: float
    llm_time: float


@dataclass
class EvaluationResult:
    """Result of evaluating a memory system on a single conversation."""

    conversation_id: ConversationID
    # Convenience flat lists (parallel to traces)
    queries: List[Prompt]
    responses: List[LLMResponse]
    # Per-turn full traces
    traces: List[QueryTrace]
    # Aggregate token counts across the whole conversation
    total_input_tokens: int
    total_output_tokens: int
    graded_input_tokens: int
    graded_output_tokens: int
    # Aggregate timing (seconds)
    total_time: float
    total_retrieval_time: float
    total_llm_time: float


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all conversations."""

    total_conversations: int
    total_queries: int
    results: List[EvaluationResult]
    # Aggregate token counts across all conversations
    total_input_tokens: int = field(default=0)
    total_output_tokens: int = field(default=0)
    total_graded_input_tokens: int = field(default=0)
    total_graded_output_tokens: int = field(default=0)
    # Aggregate timing across all conversations (seconds)
    total_time: float = field(default=0.0)
    total_retrieval_time: float = field(default=0.0)
    total_llm_time: float = field(default=0.0)
    # Per-conversation averages
    avg_input_tokens_per_conversation: float = field(default=0.0)
    avg_output_tokens_per_conversation: float = field(default=0.0)
    avg_time_per_conversation: float = field(default=0.0)
    avg_retrieval_time_per_conversation: float = field(default=0.0)
    avg_llm_time_per_conversation: float = field(default=0.0)


class Evaluator:
    """
    Evaluator for testing memory systems with different LLMs on a dataset.

    The evaluator runs through each conversation in the dataset, sending queries
    sequentially and using the memory system to retrieve relevant context.
    """

    def __init__(
        self,
        memory_system: MemorySystem,
        llm: LLM,
        dataset: ChatDataset,
        prompt_template: EvaluationPromptTemplate,
        tokenizer: EvaluationTokenizer,
    ):
        """
        Initialize the evaluator.

        Args:
            memory_system: The memory system to evaluate
            llm: The LLM to use for generating responses
            dataset: The dataset containing conversations and queries
            prompt_template: The template for formatting prompts
            tokenizer: The tokenizer to use for counting tokens
        """
        self.memory_system = memory_system
        self.llm = llm
        self.dataset = dataset
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer
        self.chat_system = ChatSystem(llm)

    def evaluate_conversation(
        self, conversation_data: ConversationData
    ) -> EvaluationResult:
        """
        Evaluate a single conversation.

        Args:
            conversation_data: The conversation data to evaluate

        Returns:
            EvaluationResult with per-turn traces and aggregate metrics
        """
        # Start a new conversation and get its ID
        conv_id = self.chat_system.start_new_conversation()

        traces: List[QueryTrace] = []
        queries: List[Prompt] = []
        responses: List[LLMResponse] = []

        conv_start_time = time.time()
        total_retrieval_time = 0.0
        total_llm_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        graded_input_tokens = 0
        graded_output_tokens = 0

        # Process each query in the conversation
        for query, should_grade in conversation_data.queries:
            # Get current conversation state
            conversation = self.chat_system.get_conversation(conv_id)
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")

            # ── Retrieval ────────────────────────────────────────────────────
            retrieval_start = time.time()
            memories = self.memory_system.get_memories(query, conversation)
            retrieval_time = time.time() - retrieval_start
            total_retrieval_time += retrieval_time

            # ── Prompt formatting ────────────────────────────────────────────
            prompt = self.prompt_template.format(query, memories, conversation)

            # Count input tokens
            input_tokens = self.tokenizer.tokenized_length(prompt)
            total_input_tokens += input_tokens
            if should_grade:
                graded_input_tokens += input_tokens

            # ── LLM call ─────────────────────────────────────────────────────
            llm_start = time.time()
            response = self.chat_system.send_message(prompt, conv_id)
            llm_time = time.time() - llm_start
            total_llm_time += llm_time

            # Count output tokens
            output_tokens = self.tokenizer.tokenized_length(response)
            total_output_tokens += output_tokens
            if should_grade:
                graded_output_tokens += output_tokens

            # ── Memory update ─────────────────────────────────────────────────
            self.memory_system.update_memory(query, response, conversation)

            # ── Record full trace for this turn ───────────────────────────────
            traces.append(
                QueryTrace(
                    query=query,
                    should_grade=should_grade,
                    retrieved_memories=memories,
                    formatted_prompt=prompt,
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retrieval_time=retrieval_time,
                    llm_time=llm_time,
                )
            )

            queries.append(query)
            responses.append(response)

        total_time = time.time() - conv_start_time

        return EvaluationResult(
            conversation_id=conv_id,
            queries=queries,
            responses=responses,
            traces=traces,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            graded_input_tokens=graded_input_tokens,
            graded_output_tokens=graded_output_tokens,
            total_time=total_time,
            total_retrieval_time=total_retrieval_time,
            total_llm_time=total_llm_time,
        )

    def evaluate(self) -> EvaluationSummary:
        """
        Evaluate the memory system on the entire dataset.

        Returns:
            EvaluationSummary with results for all conversations and aggregate metrics
        """
        results: List[EvaluationResult] = []

        for conversation_data in self.dataset:
            result = self.evaluate_conversation(conversation_data)
            results.append(result)

        n = len(results)

        # Compute aggregates across all conversations
        agg_input_tokens = sum(r.total_input_tokens for r in results)
        agg_output_tokens = sum(r.total_output_tokens for r in results)
        agg_graded_input = sum(r.graded_input_tokens for r in results)
        agg_graded_output = sum(r.graded_output_tokens for r in results)
        agg_time = sum(r.total_time for r in results)
        agg_retrieval = sum(r.total_retrieval_time for r in results)
        agg_llm = sum(r.total_llm_time for r in results)

        return EvaluationSummary(
            total_conversations=len(self.dataset),
            total_queries=self.dataset.get_total_queries(),
            results=results,
            total_input_tokens=agg_input_tokens,
            total_output_tokens=agg_output_tokens,
            total_graded_input_tokens=agg_graded_input,
            total_graded_output_tokens=agg_graded_output,
            total_time=agg_time,
            total_retrieval_time=agg_retrieval,
            total_llm_time=agg_llm,
            avg_input_tokens_per_conversation=agg_input_tokens / n if n else 0.0,
            avg_output_tokens_per_conversation=agg_output_tokens / n if n else 0.0,
            avg_time_per_conversation=agg_time / n if n else 0.0,
            avg_retrieval_time_per_conversation=agg_retrieval / n if n else 0.0,
            avg_llm_time_per_conversation=agg_llm / n if n else 0.0,
        )
