"""
Chat system implementations using the unified API.
"""

from typing import Dict, Optional
from uuid import uuid4
from .types import Conversation, ConversationID, LLM, LLMResponse, Message, Prompt


class ChatSystem:
    """
    Chat system that implements the unified API.

    This is a skeleton implementation - actual algorithm to be implemented.
    Stores conversations internally.
    """

    def __init__(self, llm: LLM):
        """
        Initialize the chat system.

        Args:
            llm: The LLM to use
        """
        self.llm = llm
        self._conversations: Dict[ConversationID, Conversation] = {}

    def start_new_conversation(self) -> ConversationID:
        """
        Start a new conversation.
        TODO: Implement actual conversation initialization.
        """
        conversation_id = uuid4()
        self._conversations[conversation_id] = Conversation(
            conversation_id=conversation_id, messages=()
        )
        return conversation_id

    def send_message(
        self, prompt: Prompt, conversation_id: ConversationID
    ) -> LLMResponse:
        """
        Send a message in a conversation.
        TODO: Implement actual message sending algorithm.
        """
        # Skeleton implementation - to be filled in with actual algorithm
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        response = self.llm.query(prompt)

        conversation = self._conversations[conversation_id]
        new_message = Message(
            prompt=prompt, response=response, conversation_id=conversation_id
        )
        updated_conversation = Conversation(
            conversation_id=conversation_id,
            messages=conversation.messages + (new_message,),
        )
        self._conversations[conversation_id] = updated_conversation

        return response

    def get_conversation(
        self, conversation_id: ConversationID
    ) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)
