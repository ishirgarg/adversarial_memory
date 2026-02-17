"""
Evaluation framework for memory systems.
"""

import time
from dataclasses import dataclass
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
class EvaluationResult:
    """Result of evaluating a memory system on a dataset."""

    conversation_id: ConversationID
    queries: List[Prompt]
    responses: List[LLMResponse]
    total_input_tokens: int
    total_output_tokens: int
    total_time: float
    total_retrieval_time: float


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all conversations."""

    total_conversations: int
    total_queries: int
    results: List[EvaluationResult]


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
            EvaluationResult with queries, responses, and total_time
        """
        # Start a new conversation and get its ID
        conv_id = self.chat_system.start_new_conversation()

        queries = []
        responses = []
        start_time = time.time()
        total_retrieval_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Process each query in the conversation
        for query in conversation_data.queries:
            # Get current conversation state
            conversation = self.chat_system.get_conversation(conv_id)
            if conversation is None:
                raise ValueError(f"Conversation {conv_id} not found")

            # Track time spent retrieving memories
            retrieval_start = time.time()
            memories = self.memory_system.get_memories(query, conversation)
            total_retrieval_time += time.time() - retrieval_start

            prompt = self.prompt_template.format(query, memories, conversation)

            # Count input tokens before sending
            total_input_tokens += self.tokenizer.tokenized_length(prompt)

            # Use ChatSystem.send_message to handle the query and update conversation
            response = self.chat_system.send_message(prompt, conv_id)

            # Count output tokens
            total_output_tokens += self.tokenizer.tokenized_length(response)

            # Update memory system with the updated conversation
            self.memory_system.update_memory(query, response, conversation)

            queries.append(query)
            responses.append(response)

        total_time = time.time() - start_time

        return EvaluationResult(
            conversation_id=conv_id,
            queries=queries,
            responses=responses,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_time=total_time,
            total_retrieval_time=total_retrieval_time,
        )

    def evaluate(self) -> EvaluationSummary:
        """
        Evaluate the memory system on the entire dataset.

        Returns:
            EvaluationSummary with results for all conversations
        """
        results = []

        for conversation_data in self.dataset:
            result = self.evaluate_conversation(conversation_data)
            results.append(result)

        return EvaluationSummary(
            total_conversations=len(self.dataset),
            total_queries=self.dataset.get_total_queries(),
            results=results,
        )
