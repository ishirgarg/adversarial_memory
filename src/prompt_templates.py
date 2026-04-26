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
        parts.append(f"User: {msg.raw_query}")
        parts.append(f"Assistant: {msg.response}")
    return "\n".join(parts)


class SimplePromptTemplate:
    """
    Simple prompt template that prepends memories to the query.
    """

    def format(self, query: Prompt, memories: str, conversation: Conversation, graded: bool = True) -> str:
        return query


class ConversationHistoryPromptTemplate:
    """
    Prompt template that includes both memories and full conversation history.

    Graded turns receive the full evaluation prompt with explicit instructions.
    Ungraded turns receive a minimal prompt (memories + history + query only).
    """

    def format(self, query: Prompt, memories: str, conversation: Conversation, graded: bool = True) -> str:
        history = format_history(conversation)

        if not graded:
            parts = [
                "You are a helpful chat assistant. Read the user's message carefully "
                "and remember any new personal information, preferences, or facts they share that you feel are important to remember."
                "They may be recalled in future conversations."
            ]
            if memories:
                parts.append(f"Relevant Past Memories:\n{memories}")
            if history:
                parts.append(f"Conversation History:\n{history}")
            parts.append(f"User: {query}")
            return "\n\n".join(parts)

        return f"""You are an intelligent memory assistant tasked with answering questions using information from past conversation memories.

# CONTEXT:
You have access to memories from previous conversations as well as the conversation history that may be helpful in answering the question.

# INSTRUCTIONS:
Answer the user's question. You may use the provided memories if they are helpful. If you use the memories above to answer a question, please EXPLICITLY RESTATE which memories you used below, or state that you used no memories. You should only use and restate those memories if you explicitly used them to draw conclusions from them.
Conversation History:
{history}
Relevant Memories:
{memories}
END of Relevant Memories

User Question:
{query}
Answer:"""
