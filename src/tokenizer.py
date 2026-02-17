"""
Tokenizer for evaluation metrics.
"""

from typing import Protocol, runtime_checkable
import tiktoken


@runtime_checkable
class EvaluationTokenizer(Protocol):
    """
    Protocol for tokenizers used in evaluation.

    Tokenizers convert text to tokens and can count tokenized length.
    """

    def tokenized_length(self, text: str) -> int:
        """
        Get the tokenized length of a string.

        Args:
            text: The text to tokenize

        Returns:
            The number of tokens in the tokenized text
        """
        ...


class TiktokenTokenizer:
    """
    Tokenizer using tiktoken (OpenAI's tokenizer).

    Uses the cl100k_base encoding by default (used by GPT-3.5 and GPT-4).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the tiktoken tokenizer.

        Args:
            encoding_name: The encoding to use (default: "cl100k_base")
                Common options:
                - "cl100k_base": GPT-3.5, GPT-4
                - "p50k_base": Codex, GPT-3
                - "r50k_base": GPT-3 (legacy)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def tokenized_length(self, text: str) -> int:
        """
        Get the tokenized length of a string.

        Args:
            text: The text to tokenize

        Returns:
            The number of tokens in the tokenized text
        """
        return len(self.encoding.encode(text))
