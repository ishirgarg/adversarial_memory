"""
Dataset classes for evaluating memory systems.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .types import Prompt


@dataclass
class ConversationData:
    """
    Represents a conversation in the dataset, and booleans indicating whether the query should be graded - responses will be generated during evaluation.
    """

    queries: List[Tuple[Prompt, bool]]

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
                "queries": [
                    {"query": "query1", "grade": true},
                    {"query": "query2", "grade": false}
                ]
            }
        ]
    }

    Each query must be an object with "query" (string) and "grade" (boolean) fields.
    The "grade" field indicates whether the response should be graded.

    Conversation IDs are assigned during evaluation by the evaluator.
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
            KeyError: If queries or grade are missing
            ValueError: If query format is invalid

        The dataset format requires each query to be an object with "query" and "grade":
        {
            "conversations": [
                {
                    "queries": [
                        {"query": "query1", "grade": true},
                        {"query": "query2", "grade": false}
                    ]
                }
            ]
        }
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = []
        for conv_data in data["conversations"]:
            query_tuples = []
            for query_item in conv_data["queries"]:
                if not isinstance(query_item, dict):
                    raise ValueError(
                        f"Query must be an object with 'query' and 'grade' fields. "
                        f"Got: {type(query_item).__name__}"
                    )
                
                if "query" not in query_item:
                    raise KeyError("Query object must have 'query' field")
                if "grade" not in query_item:
                    raise KeyError("Query object must have 'grade' field")
                
                query = query_item["query"]
                grade = bool(query_item["grade"])
                query_tuples.append((query, grade))
            
            conversations.append(ConversationData(queries=query_tuples))

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
            KeyError: If queries or grade are missing
            ValueError: If query format is invalid

        The dataset format requires each query to be an object with "query" and "grade":
        {
            "conversations": [
                {
                    "queries": [
                        {"query": "query1", "grade": true},
                        {"query": "query2", "grade": false}
                    ]
                }
            ]
        }
        """
        conversations = []
        for conv_data in data["conversations"]:
            query_tuples = []
            for query_item in conv_data["queries"]:
                if not isinstance(query_item, dict):
                    raise ValueError(
                        f"Query must be an object with 'query' and 'grade' fields. "
                        f"Got: {type(query_item).__name__}"
                    )
                
                if "query" not in query_item:
                    raise KeyError("Query object must have 'query' field")
                if "grade" not in query_item:
                    raise KeyError("Query object must have 'grade' field")
                
                query = query_item["query"]
                grade = bool(query_item["grade"])
                query_tuples.append((query, grade))
            
            conversations.append(ConversationData(queries=query_tuples))

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
