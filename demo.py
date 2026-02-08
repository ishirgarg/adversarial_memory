#!/usr/bin/env python3
"""
Demo using Ollama gemma2:1b with NoHistoryMemorySystem.
"""

from src import OllamaLLM, ChatSystem, NoHistoryMemorySystem


def main():
    print("=" * 60)
    print("Ollama Demo with NoHistoryMemorySystem")
    print("=" * 60)

    # Create LLM (using gemma3:1b)
    print("\n1. Initializing Ollama LLM (gemma3:1b)...")
    llm = OllamaLLM(model="gemma3:1b")

    # Create memory system
    print("2. Creating NoHistoryMemorySystem...")
    _memory = NoHistoryMemorySystem()

    # Create chat system
    print("3. Creating ChatSystem...")
    chat = ChatSystem(llm)

    # Start a conversation
    print("\n4. Starting new conversation...")
    conv_id = chat.start_new_conversation()
    print(f"   Conversation ID: {conv_id}")

    # Send a message
    print("\n5. Sending message...")
    prompt = "What is the capital of France?"
    print(f"   Prompt: {prompt}")

    try:
        response = chat.send_message(prompt, conv_id)
        print(f"\n   Response: {response}")

        # Show conversation history
        print("\n6. Conversation history:")
        conversation = chat.get_conversation(conv_id)
        if conversation:
            for i, msg in enumerate(conversation.messages, 1):
                print(f"   Message {i}:")
                print(f"     User: {msg.prompt}")
                print(f"     Assistant: {msg.response[:100]}...")

    except Exception as e:
        print(f"\n   Error: {e}")
        print("   Make sure Ollama is running and gemma3:1b model is available:")
        print("   - Install Ollama: https://ollama.ai")
        print("   - Pull model: ollama pull gemma3:1b")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
