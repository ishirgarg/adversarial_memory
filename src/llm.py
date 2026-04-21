"""
LLM implementations for the framework.

Provides simple LLM implementations that follow the LLM protocol.
"""

from typing import Any, Optional
from .types import LLMResponse, LLMUsage
import requests
from openai import OpenAI

# Cost per 1M tokens: (input_price, output_price) in USD
# Keys ordered most-specific first so startswith matching is unambiguous
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # gpt-5.4 family
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.4": (2.50, 15.00),
    # gpt-5.2 family
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5.2": (1.75, 14.00),
    # gpt-5 family
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    # gpt-4.1 family
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    # gpt-4o family
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # o-series
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for one OpenAI API call. Returns 0.0 for unknown models."""
    for prefix, (input_price, output_price) in OPENAI_PRICING.items():
        if model.startswith(prefix):
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return 0.0


class OpenAILLM:
    """
    OpenAI LLM implementation.
    Skeleton - to be implemented with actual API calls.
    """

    def __init__(self, api_key: str, model: str, **default_kwargs: Any):
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
        self.last_usage: Optional[LLMUsage] = None

    def query(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Query OpenAI API.
        """
        client = OpenAI(api_key=self.api_key)

        # Merge default kwargs with provided kwargs
        request_kwargs = {**self.default_kwargs, **kwargs}

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **request_kwargs,
        )

        if response.usage is not None:
            self.last_usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        else:
            self.last_usage = None

        return response.choices[0].message.content or ""


class AnthropicLLM:
    """
    Anthropic Claude LLM implementation.
    Skeleton - to be implemented with actual API calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
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
        model: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
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
