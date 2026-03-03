"""
chat.py — Minimalist ChatGPT-style CLI interface.

A simple interactive chat interface that supports:
- Multiple memory systems (NoHistory, SimpleHistory, Mem0, A-MEM)
- Creating new conversations
- Viewing and continuing previous conversations
- Memory retrieval and updates

Usage:
  uv run app/chat.py --memory nohistory --llm openai --model gpt-4o-mini
  uv run app/chat.py --memory mem0 --num-memories 5 --llm ollama --model llama2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    AMEMMemorySystem,
    ChatSystem,
    ConversationHistoryPromptTemplate,
    Mem0MemorySystem,
    NoHistoryMemorySystem,
    OpenAILLM,
    OllamaLLM,
    SimpleHistoryMemorySystem,
    SimplePromptTemplate,
)
from src.types import Conversation, ConversationID

load_dotenv()


# ---------------------------------------------------------------------------
# Memory system factory
# ---------------------------------------------------------------------------

def create_memory_system(memory_type: str, **kwargs):
    """Create a memory system based on type."""
    if memory_type == "nohistory":
        return NoHistoryMemorySystem()
    elif memory_type == "simple":
        return SimpleHistoryMemorySystem()
    elif memory_type == "mem0":
        num_memories = kwargs.pop("num_memories", 5)
        shared_user_id = kwargs.pop("shared_user_id", None)
        return Mem0MemorySystem(num_memories=num_memories, shared_user_id=shared_user_id, **kwargs)
    elif memory_type == "amem":
        return AMEMMemorySystem(
            num_memories=kwargs.pop("num_memories", 5),
            llm_backend=kwargs.pop("llm_backend", "openai"),
            llm_model=kwargs.pop("llm_model", "gpt-4o-mini"),
            embedding_model=kwargs.pop("embedding_model", "all-MiniLM-L6-v2"),
            evo_threshold=kwargs.pop("evo_threshold", 10),
            api_key=kwargs.pop("api_key", os.getenv("OPENAI_KEY")),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown memory system: {memory_type}")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def create_llm(llm_type: str, **kwargs):
    """Create an LLM based on type."""
    if llm_type == "openai":
        api_key = kwargs.pop("api_key", os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("OPENAI_KEY or OPENAI_API_KEY environment variable required")
        model = kwargs.pop("model", "gpt-4o-mini")
        return OpenAILLM(api_key=api_key, model=model, **kwargs)
    elif llm_type == "ollama":
        model = kwargs.pop("model", "llama2")
        base_url = kwargs.pop("base_url", "http://localhost:11434")
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        return OllamaLLM(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

class ChatInterface:
    """Minimalist chat interface."""

    def __init__(
        self,
        memory_system,
        chat_system: ChatSystem,
        prompt_template,
        current_conv_id: Optional[ConversationID] = None,
    ):
        self.memory_system = memory_system
        self.chat_system = chat_system
        self.prompt_template = prompt_template
        self.current_conv_id = current_conv_id

    def print_conversation(self, conv_id: ConversationID):
        """Print a conversation's history."""
        conv = self.chat_system.get_conversation(conv_id)
        if not conv:
            print("  (Conversation not found)")
            return

        print(f"\n  Conversation {str(conv_id)[:8]}...")
        if not conv.messages:
            print("  (No messages yet)")
        else:
            for i, msg in enumerate(conv.messages, 1):
                print(f"\n  [{i}] User: {msg.prompt}")
                print(f"      Assistant: {msg.response[:200]}{'...' if len(msg.response) > 200 else ''}")

    def list_conversations(self):
        """List all conversations."""
        # ChatSystem stores conversations internally, but doesn't expose a list method
        # For now, we'll just show the current conversation if it exists
        if self.current_conv_id:
            print("\n=== Current Conversation ===")
            self.print_conversation(self.current_conv_id)
        else:
            print("\n(No active conversation)")

    def start_new_conversation(self) -> ConversationID:
        """Start a new conversation."""
        conv_id = self.chat_system.start_new_conversation()
        self.current_conv_id = conv_id
        print(f"\n✓ Started new conversation: {str(conv_id)[:8]}...")
        return conv_id

    def continue_conversation(self, conv_id_str: str) -> bool:
        """Continue an existing conversation."""
        try:
            conv_id = UUID(conv_id_str)
        except ValueError:
            print(f"Invalid conversation ID: {conv_id_str}")
            return False

        conv = self.chat_system.get_conversation(conv_id)
        if not conv:
            print(f"Conversation {conv_id_str} not found")
            return False

        self.current_conv_id = conv_id
        print(f"\n✓ Switched to conversation: {str(conv_id)[:8]}...")
        self.print_conversation(conv_id)
        return True

    def send_message(self, prompt: str) -> Optional[str]:
        """Send a message in the current conversation."""
        if not self.current_conv_id:
            print("No active conversation. Start a new one first.")
            return None

        try:
            # Get current conversation state
            conversation = self.chat_system.get_conversation(self.current_conv_id)
            if not conversation:
                print("Conversation not found")
                return None

            # Retrieve memories
            memories = self.memory_system.get_memories(prompt, conversation)

            # Format prompt
            formatted_prompt = self.prompt_template.format(prompt, memories, conversation)

            # Send message
            print("\n[Thinking...]")
            response = self.chat_system.send_message(formatted_prompt, self.current_conv_id)

            # Update memory
            updated_conversation = self.chat_system.get_conversation(self.current_conv_id)
            if updated_conversation:
                self.memory_system.update_memory(prompt, response, updated_conversation)

            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def run(self):
        """Run the interactive chat loop."""
        print("\n" + "=" * 60)
        print("Chat Interface")
        print("=" * 60)
        print("\nCommands:")
        print("  /new          - Start a new conversation")
        print("  /list         - List conversations")
        print("  /continue ID  - Continue a conversation by ID")
        print("  /quit         - Quit")
        print("\nType a message to chat, or use a command above.\n")

        if not self.current_conv_id:
            print("Starting a new conversation...")
            self.start_new_conversation()

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0]

                    if cmd == "/quit":
                        print("\nGoodbye!")
                        break
                    elif cmd == "/new":
                        self.start_new_conversation()
                    elif cmd == "/list":
                        self.list_conversations()
                    elif cmd == "/continue" and len(parts) > 1:
                        self.continue_conversation(parts[1])
                    else:
                        print(f"Unknown command: {cmd}")
                    continue

                # Send message
                response = self.send_message(user_input)
                if response:
                    print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimalist ChatGPT-style CLI interface")
    
    # Memory system
    parser.add_argument(
        "--memory",
        choices=["nohistory", "simple", "mem0", "amem"],
        default="nohistory",
        help="Memory system to use (default: nohistory)",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=5,
        help="Number of memories to retrieve (for mem0/amem, default: 5)",
    )
    parser.add_argument(
        "--shared-user-id",
        type=str,
        default=None,
        help="Shared user ID for mem0 (default: None, uses conversation ID)",
    )
    
    # A-MEM specific
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM backend for A-MEM (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for A-MEM (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for A-MEM (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--evo-threshold",
        type=int,
        default=10,
        help="Evolution threshold for A-MEM (default: 10)",
    )
    
    # LLM
    parser.add_argument(
        "--llm",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for Ollama (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature (default: 0.7)",
    )
    
    # Prompt template
    parser.add_argument(
        "--prompt-template",
        choices=["simple", "conversation"],
        default="conversation",
        help="Prompt template (default: conversation)",
    )

    args = parser.parse_args()

    # Create memory system
    print(f"Initializing {args.memory} memory system...")
    memory_kwargs = {"num_memories": args.num_memories}
    if args.shared_user_id:
        memory_kwargs["shared_user_id"] = args.shared_user_id
    if args.memory == "amem":
        memory_kwargs.update({
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "evo_threshold": args.evo_threshold,
        })
    
    try:
        memory_system = create_memory_system(args.memory, **memory_kwargs)
    except Exception as e:
        sys.exit(f"Error creating memory system: {e}")

    # Create LLM
    print(f"Initializing {args.llm} LLM ({args.model})...")
    llm_kwargs = {"model": args.model}
    if args.llm == "ollama":
        llm_kwargs.update({
            "base_url": args.ollama_base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        })
    
    try:
        llm = create_llm(args.llm, **llm_kwargs)
    except Exception as e:
        sys.exit(f"Error creating LLM: {e}")

    # Create chat system
    chat_system = ChatSystem(llm)

    # Create prompt template
    if args.prompt_template == "simple":
        prompt_template = SimplePromptTemplate()
    else:
        prompt_template = ConversationHistoryPromptTemplate()

    # Create and run interface
    interface = ChatInterface(memory_system, chat_system, prompt_template)
    interface.run()


if __name__ == "__main__":
    main()
