"""
Dataset classes for evaluating memory systems.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .types import Prompt


@dataclass
class ConversationData:
    """
    Represents a conversation in the dataset.

    Contains only user queries - responses will be generated during evaluation.
    """

    queries: List[Prompt]

    def __len__(self) -> int:
        """Return the number of queries in this conversation."""
        return len(self.queries)


class ChatDataset:
    """
    Dataset representing multiple conversations with user queries.

    The dataset format is JSON with the following structure:
    {
        "conversations": [
            {
                "queries": ["query1", "query2", "query3"]
            },
            {
                "queries": ["query1", "query2"]
            }
        ]
    }

    Each conversation is just a list of queries. Conversation IDs are assigned
    during evaluation by the evaluator.
    """

    def __init__(self, conversations: List[ConversationData]):
        """
        Initialize the dataset.

        Args:
            conversations: List of conversation data
        """
        self.conversations = conversations

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ChatDataset":
        """
        Load dataset from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            ChatDataset instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If queries are missing
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = []
        for conv_data in data["conversations"]:
            queries = conv_data["queries"]
            conversations.append(ConversationData(queries=queries))

        return cls(conversations)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatDataset":
        """
        Create dataset from a dictionary.

        Args:
            data: Dictionary with "conversations" key containing list of conversation dicts

        Returns:
            ChatDataset instance

        Raises:
            KeyError: If queries are missing
        """
        conversations = []
        for conv_data in data["conversations"]:
            queries = conv_data["queries"]
            conversations.append(ConversationData(queries=queries))

        return cls(conversations)

    def __len__(self) -> int:
        """Return the number of conversations in the dataset."""
        return len(self.conversations)

    def __iter__(self):
        """Iterate over conversations in the dataset."""
        return iter(self.conversations)

    def get_total_queries(self) -> int:
        """Return the total number of queries across all conversations."""
        return sum(len(conv) for conv in self.conversations)
