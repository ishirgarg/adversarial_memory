"""
Memory system implementations for LLM conversations.
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
import requests

from .types import Conversation, LLMResponse, Prompt


class NoHistoryMemorySystem:
    """Baseline that ignores conversation history and returns empty context."""

    def __init__(self, num_memories: int | None = None):
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        return ""

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        pass

    def get_all_memories(self) -> list[str]:
        return []

    def finalize_conversation(self, conversation: Conversation) -> None:
        pass


class SimpleHistoryMemorySystem:
    """Prepends raw conversation history to the prompt. Stores no memory internally."""

    def __init__(self, num_memories: int | None = None):
        self.num_memories = num_memories

    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        if not conversation.messages:
            return ""

        messages_to_include = conversation.messages
        if self.num_memories is not None:
            max_messages = self.num_memories * 2
            if len(conversation.messages) > max_messages:
                messages_to_include = conversation.messages[-max_messages:]

        history_parts = []
        for msg in messages_to_include:
            history_parts.append(f"User: {msg.prompt}")
            history_parts.append(f"Assistant: {msg.response}")

        return "\n".join(history_parts)

    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        pass

    def get_all_memories(self) -> list[str]:
        return []

    def finalize_conversation(self, conversation: Conversation) -> None:
        pass

