"""
LLM implementations for the framework.

Provides simple LLM implementations that follow the LLM protocol.
"""

from typing import Any
from .types import LLMResponse
import requests


class OpenAILLM:
    """
    OpenAI LLM implementation.
    Skeleton - to be implemented with actual API calls.
    """

    def __init__(
        self, api_key: str, model: str = "gpt-3.5-turbo", **default_kwargs: Any
    ):
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            model: Model name
            **default_kwargs: Default parameters for API calls
        """
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs

    def query(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Query OpenAI API.
        TODO: Implement actual OpenAI API call.
        """
        # Skeleton implementation
        raise NotImplementedError("OpenAI API implementation pending")


class AnthropicLLM:
    """
    Anthropic Claude LLM implementation.
    Skeleton - to be implemented with actual API calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        **default_kwargs: Any,
    ):
        """
        Initialize Anthropic LLM.

        Args:
            api_key: Anthropic API key
            model: Model name
            **default_kwargs: Default parameters for API calls
        """
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs

    def query(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Query Anthropic API.
        TODO: Implement actual Anthropic API call.
        """
        # Skeleton implementation
        raise NotImplementedError("Anthropic API implementation pending")


class OllamaLLM:
    """
    Ollama LLM implementation.
    """

    def __init__(
        self,
        model: str = "gemma2:1b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **default_kwargs: Any,
    ):
        """
        Initialize Ollama LLM.

        Args:
            model: Model name (e.g., "gemma2:1b", "llama2", etc.)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            max_tokens: Maximum number of tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.7)
            **default_kwargs: Additional default parameters for API calls
        """
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_kwargs = default_kwargs

    def query(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Query Ollama API.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            **self.default_kwargs,
            **kwargs,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")
