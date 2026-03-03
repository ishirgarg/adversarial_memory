"""
Prompt templates for evaluation.
"""

from .types import Conversation, Prompt


def format_history(conversation: Conversation) -> str:
    """
    Format the conversation history.
    """
    parts = []
    for msg in conversation.messages:
        parts.append(f"User: {msg.prompt}")
        parts.append(f"Assistant: {msg.response}")
    return "\n".join(parts)


class SimplePromptTemplate:
    """
    Simple prompt template that prepends memories to the query.
    """

    def format(self, query: Prompt, memories: str, conversation: Conversation) -> str:
        """
        Format the prompt by prepending memories to the query.

        Args:
            query: The current user query
            memories: The memories retrieved from the memory system
            conversation: The conversation history (not used in this template)

        Returns:
            The formatted prompt string
        """
        return query


class ConversationHistoryPromptTemplate:
    """
    Prompt template that includes both memories and full conversation history.
    """

    def format(self, query: Prompt, memories: str, conversation: Conversation) -> str:
        """
        Format the prompt with memories and conversation history.

        Args:
            query: The current user query
            memories: The memories retrieved from the memory system
            conversation: The conversation history

        Returns:
            The formatted prompt string
        """

        prompt = f"""You are an intelligent memory assistant tasked with answering questions using information from past conversation memories.

# CONTEXT:
You have access to memories from previous conversations as well as the conversation history.

# INSTRUCTIONS:
Answer the user's question. You may use the provided memories if they are helpful. If the user does not ask a question, you do not need to respond nor verify the statement as if it were a question, just respond as normal and update your memory with the new information.
Conversation History:
{format_history(conversation)}
Relevant Memories:
{memories}
END of Relevant Memories

User Question:
{query}
Answer:"""
        return prompt
