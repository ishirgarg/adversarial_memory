"""
chat_ui.py — Minimalist ChatGPT-style web UI using Streamlit.

A simple interactive chat interface that supports:
- Multiple memory systems (NoHistory, SimpleHistory, Mem0, A-MEM, SimpleMem)
- Creating new conversations
- Viewing and continuing previous conversations
- Memory retrieval and updates

Usage:
  streamlit run app/chat_ui.py -- --memory nohistory --llm openai --model gpt-4o-mini
  streamlit run app/chat_ui.py -- --memory mem0 --num-memories 5
  streamlit run app/chat_ui.py -- --memory simplemem --model gpt-4o-mini
  streamlit run app/chat_ui.py -- --memory simplemem --simplemem-db-path ./my_memory_db
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

import streamlit as st
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
    SimpleMemMemorySystem,
    SimplePromptTemplate,
)
from src.types import ConversationID

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
        # Always use shared user ID for mem0 in chat UI
        shared_user_id = kwargs.pop("shared_user_id", "shared_user")
        embedding_provider = kwargs.pop("embedding_provider", None)
        embedding_model = kwargs.pop("embedding_model", None)
        return Mem0MemorySystem(
            num_memories=num_memories,
            shared_user_id=shared_user_id,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            **kwargs,
        )
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
    elif memory_type == "simplemem":
        return SimpleMemMemorySystem(
            api_key=kwargs.pop("api_key", os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")),
            model=kwargs.pop("model", None),
            base_url=kwargs.pop("base_url", None),
            db_path=kwargs.pop("db_path", None),
            clear_db=kwargs.pop("clear_db", False),
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
# Streamlit UI
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_system" not in st.session_state:
        st.session_state.chat_system = None
    if "memory_system" not in st.session_state:
        st.session_state.memory_system = None
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = None
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # Store conversation metadata
    if "original_queries" not in st.session_state:
        st.session_state.original_queries = {}  # Store original user queries by conv_id
    if "formatted_prompts" not in st.session_state:
        st.session_state.formatted_prompts = {}  # Store formatted prompts by conv_id
    if "show_all_memories" not in st.session_state:
        st.session_state.show_all_memories = False


def send_message(prompt: str) -> Optional[str]:
    """Send a message in the current conversation."""
    if not st.session_state.current_conv_id:
        return None

    try:
        # Get current conversation state
        conversation = st.session_state.chat_system.get_conversation(st.session_state.current_conv_id)
        if not conversation:
            return None

        # Store original query
        conv_id_str = str(st.session_state.current_conv_id)
        if conv_id_str not in st.session_state.original_queries:
            st.session_state.original_queries[conv_id_str] = []
        st.session_state.original_queries[conv_id_str].append(prompt)

        # Create a conversation with original queries for history formatting
        from src.types import Message, Conversation
        original_messages = []
        if conv_id_str in st.session_state.original_queries:
            # Get original queries excluding the current one we just added
            original_queries = st.session_state.original_queries[conv_id_str][:-1]  # Exclude current
            # Match with existing conversation messages (should be same length)
            for i, msg in enumerate(conversation.messages):
                if i < len(original_queries):
                    # Use original query if available
                    original_messages.append(Message(
                        prompt=original_queries[i],  # Use original query, not formatted prompt
                        response=msg.response,
                        conversation_id=msg.conversation_id
                    ))
                else:
                    # Fallback: use the stored prompt (shouldn't happen normally)
                    original_messages.append(msg)
        
        history_conversation = Conversation(
            conversation_id=conversation.conversation_id,
            messages=tuple(original_messages)
        )

        # Retrieve memories using history with original queries
        memories = st.session_state.memory_system.get_memories(prompt, history_conversation)

        # Format prompt using history with original queries
        formatted_prompt = st.session_state.prompt_template.format(prompt, memories, history_conversation)

        # Store formatted prompt for display
        conv_id_str = str(st.session_state.current_conv_id)
        if conv_id_str not in st.session_state.formatted_prompts:
            st.session_state.formatted_prompts[conv_id_str] = []
        st.session_state.formatted_prompts[conv_id_str].append(formatted_prompt)

        # Send message (store formatted prompt in chat system, but we track originals separately)
        response = st.session_state.chat_system.send_message(formatted_prompt, st.session_state.current_conv_id)

        # Update memory using history with original queries
        updated_conversation = st.session_state.chat_system.get_conversation(st.session_state.current_conv_id)
        if updated_conversation:
            # Rebuild with original queries for memory update
            updated_original_messages = []
            if conv_id_str in st.session_state.original_queries:
                updated_original_queries = st.session_state.original_queries[conv_id_str]
                # Match with updated conversation messages
                for i, msg in enumerate(updated_conversation.messages):
                    if i < len(updated_original_queries):
                        updated_original_messages.append(Message(
                            prompt=updated_original_queries[i],
                            response=msg.response,
                            conversation_id=msg.conversation_id
                        ))
                    else:
                        # Fallback: use stored prompt
                        updated_original_messages.append(msg)
            else:
                # No original queries stored, use messages as-is
                updated_original_messages = list(updated_conversation.messages)
            
            updated_history_conversation = Conversation(
                conversation_id=updated_conversation.conversation_id,
                messages=tuple(updated_original_messages)
            )
            st.session_state.memory_system.update_memory(prompt, response, updated_history_conversation)

        return response
    except Exception as e:
        import traceback
        st.error(f"Error in send_message: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return None


def start_new_conversation():
    """Start a new conversation."""
    conv_id = st.session_state.chat_system.start_new_conversation()
    st.session_state.current_conv_id = conv_id
    conv_id_str = str(conv_id)
    st.session_state.conversations[conv_id_str] = {
        "id": conv_id,
        "title": "New Conversation",
        "created": True,
    }
    # Initialize original queries and formatted prompts lists for this conversation
    st.session_state.original_queries[conv_id_str] = []
    st.session_state.formatted_prompts[conv_id_str] = []
    return conv_id


def main():
    st.set_page_config(page_title="Chat Interface", page_icon="💬", layout="wide")

    # Parse command line arguments
    if "args" not in st.session_state:
        # Parse args from sys.argv (Streamlit passes them after --)
        args_list = sys.argv[1:]
        if "--" in args_list:
            args_list = args_list[args_list.index("--") + 1:]
        
        parser = argparse.ArgumentParser(description="Chat UI")
        parser.add_argument("--memory", choices=["nohistory", "simple", "mem0", "amem", "simplemem"], default="nohistory")
        parser.add_argument("--num-memories", type=int, default=5)
        parser.add_argument("--llm-backend", choices=["openai", "ollama"], default="openai")
        parser.add_argument("--llm-model", default="gpt-4o-mini")
        parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name (for mem0 or amem)")
        parser.add_argument("--embedding-provider", choices=["openai", "ollama", "sentence-transformers"], default=None, help="Embedding provider for mem0 (default: mem0's default)")
        parser.add_argument("--evo-threshold", type=int, default=10)
        parser.add_argument("--llm", choices=["openai", "ollama"], default="openai")
        parser.add_argument("--model", default="gpt-4o-mini")
        parser.add_argument("--ollama-base-url", default="http://localhost:11434")
        parser.add_argument("--max-tokens", type=int, default=512)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--prompt-template", choices=["simple", "conversation"], default="conversation")
        parser.add_argument("--simplemem-db-path", default=None, help="LanceDB path for SimpleMem (default: SimpleMem config default)")
        parser.add_argument("--simplemem-clear-db", action="store_true", help="Clear SimpleMem database on startup (default: keep existing memories)")
        
        try:
            st.session_state.args = parser.parse_args(args_list)
        except SystemExit:
            st.session_state.args = parser.parse_args([])  # Use defaults

    args = st.session_state.args

    initialize_session_state()

    # Initialize systems if not already done
    if st.session_state.chat_system is None:
        with st.spinner("Initializing systems..."):
            try:
                # Create memory system
                if args.memory == "mem0":
                    memory_kwargs = {"num_memories": args.num_memories}
                    if args.embedding_provider and args.embedding_model:
                        memory_kwargs["embedding_provider"] = args.embedding_provider
                        memory_kwargs["embedding_model"] = args.embedding_model
                        if args.embedding_provider == "ollama":
                            memory_kwargs["ollama_base_url"] = args.ollama_base_url
                elif args.memory == "amem":
                    memory_kwargs = {
                        "num_memories": args.num_memories,
                        "llm_backend": args.llm_backend,
                        "llm_model": args.llm_model,
                        "embedding_model": args.embedding_model,
                        "evo_threshold": args.evo_threshold,
                    }
                elif args.memory == "simplemem":
                    memory_kwargs = {
                        "model": args.model,
                        "db_path": args.simplemem_db_path,
                        "clear_db": args.simplemem_clear_db,
                    }
                else:
                    memory_kwargs = {}

                st.session_state.memory_system = create_memory_system(args.memory, **memory_kwargs)

                # Create LLM
                llm_kwargs = {"model": args.model}
                if args.llm == "ollama":
                    llm_kwargs.update({
                        "base_url": args.ollama_base_url,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                    })
                
                st.session_state.llm = create_llm(args.llm, **llm_kwargs)

                # Create chat system
                st.session_state.chat_system = ChatSystem(st.session_state.llm)

                # Create prompt template
                if args.prompt_template == "simple":
                    st.session_state.prompt_template = SimplePromptTemplate()
                else:
                    st.session_state.prompt_template = ConversationHistoryPromptTemplate()

                # Start initial conversation
                if st.session_state.current_conv_id is None:
                    start_new_conversation()

                st.success("Systems initialized!")
            except Exception as e:
                st.error(f"Error initializing: {e}")
                st.stop()

    # Sidebar for conversation management
    with st.sidebar:
        st.header("💬 Conversations")
        
        if st.button("➕ New Conversation", use_container_width=True):
            start_new_conversation()
            st.rerun()

        st.divider()
        
        # List conversations
        st.subheader("Conversations")
        for conv_id_str, conv_data in st.session_state.conversations.items():
            conv_id = conv_data["id"]
            is_active = st.session_state.current_conv_id == conv_id
            
            # Get conversation to show message count
            conv = st.session_state.chat_system.get_conversation(conv_id)
            msg_count = len(conv.messages) if conv else 0
            
            label = f"{'▶️' if is_active else '○'} {str(conv_id)[:8]}... ({msg_count} msgs)"
            
            if st.button(label, key=f"conv_{conv_id_str}", use_container_width=True):
                st.session_state.current_conv_id = conv_id
                # Initialize original_queries and formatted_prompts if not present
                if conv_id_str not in st.session_state.original_queries:
                    st.session_state.original_queries[conv_id_str] = []
                if conv_id_str not in st.session_state.formatted_prompts:
                    st.session_state.formatted_prompts[conv_id_str] = []
                st.rerun()

        st.divider()
        
        # Show all memories button for mem0 / simplemem
        if args.memory in ("mem0", "simplemem") and st.session_state.memory_system:
            if st.button("📚 View All Memories", use_container_width=True):
                st.session_state.show_all_memories = True
                st.rerun()
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        st.text(f"Memory: {args.memory}")
        st.text(f"LLM: {args.llm} ({args.model})")
        if args.memory in ["mem0", "amem"]:
            st.text(f"Memories: {args.num_memories}")
        if args.memory == "simplemem":
            db_label = args.simplemem_db_path or "default"
            st.text(f"DB: {db_label}")
            st.text(f"Persist: {'no (cleared)' if args.simplemem_clear_db else 'yes'}")

    # Show all memories modal/dialog
    if st.session_state.show_all_memories and args.memory == "mem0" and st.session_state.memory_system:
        st.title("📚 All Memories")
        st.caption("All memories for shared_user")
        
        try:
            # Get all memories for the shared user
            shared_user_id = st.session_state.memory_system.shared_user_id or "shared_user"
            
            # Try to get all memories using mem0's API
            # mem0.Memory might have a get_all or similar method
            # If not available, we'll try accessing the underlying client
            memory_obj = st.session_state.memory_system.memory
            
            # Check if memory object has get_all method or client attribute
            if hasattr(memory_obj, 'get_all'):
                # Try with just user_id parameter first (no filters for all memories)
                try:
                    all_memories = memory_obj.get_all(user_id=shared_user_id)
                except Exception:
                    # If that fails, try with empty filters
                    all_memories = memory_obj.get_all(
                        user_id=shared_user_id,
                        filters={}
                    )
            elif hasattr(memory_obj, 'client') and hasattr(memory_obj.client, 'get_all'):
                # Try with just user_id parameter first
                try:
                    all_memories = memory_obj.client.get_all(user_id=shared_user_id)
                except Exception:
                    # If that fails, try with empty filters
                    all_memories = memory_obj.client.get_all(
                        user_id=shared_user_id,
                        filters={}
                    )
            else:
                # Fallback: try to search with a very broad query and high limit
                # This is not ideal but works if get_all is not available
                st.warning("Direct get_all not available, using search fallback")
                search_result = memory_obj.search(query="", user_id=shared_user_id, limit=1000)
                all_memories = {"results": search_result.get("results", [])}
            
            if all_memories and "results" in all_memories and len(all_memories["results"]) > 0:
                st.success(f"Found {len(all_memories['results'])} memories")
                
                for i, memory in enumerate(all_memories["results"], 1):
                    with st.expander(f"Memory {i}", expanded=False):
                        if isinstance(memory, dict):
                            # Display all fields in the memory
                            for key, value in memory.items():
                                if key != "id":  # Skip ID for cleaner display
                                    st.text(f"**{key}**: {value}")
                        else:
                            st.write(memory)
            else:
                st.info("No memories found")
                
        except Exception as e:
            st.error(f"Error retrieving memories: {e}")
            st.code(str(e))
        
        if st.button("Close", use_container_width=True):
            st.session_state.show_all_memories = False
            st.rerun()
        
        st.divider()
    
    # SimpleMem — View All Memories panel
    if st.session_state.show_all_memories and args.memory == "simplemem" and st.session_state.memory_system:
        st.title("📚 SimpleMem — All Memories")
        st.caption("All memory entries stored in LanceDB across conversations")

        try:
            table = st.session_state.memory_system._system.vector_store.table
            df = table.to_pandas()

            if not df.empty:
                st.success(f"Found {len(df)} memory entries")
                for i, row in df.iterrows():
                    label = str(row.get("lossless_restatement", ""))[:80] or f"Entry {i + 1}"
                    trailing = "..." if len(str(row.get("lossless_restatement", ""))) > 80 else ""
                    with st.expander(f"[{i + 1}] {label}{trailing}", expanded=False):
                        st.markdown(f"**Memory:** {row.get('lossless_restatement', '—')}")
                        if row.get("topic"):
                            st.markdown(f"**Topic:** {row['topic']}")
                        kw = row.get("keywords")
                        if kw is not None and len(kw) > 0:
                            st.markdown(f"**Keywords:** {', '.join(kw)}")
                        persons = row.get("persons")
                        if persons is not None and len(persons) > 0:
                            st.markdown(f"**Persons:** {', '.join(persons)}")
                        entities = row.get("entities")
                        if entities is not None and len(entities) > 0:
                            st.markdown(f"**Entities:** {', '.join(entities)}")
                        if row.get("location"):
                            st.markdown(f"**Location:** {row['location']}")
                        if row.get("timestamp"):
                            st.markdown(f"**Timestamp:** {row['timestamp']}")
            else:
                st.info("No memories stored yet. Start a conversation to build memory.")
        except Exception as e:
            import traceback
            st.error(f"Error reading SimpleMem database: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

        if st.button("Close", use_container_width=True):
            st.session_state.show_all_memories = False
            st.rerun()

        st.divider()

    # Main chat area
    st.title("💬 Chat Interface")
    
    if st.session_state.current_conv_id:
        st.caption(f"Conversation: {str(st.session_state.current_conv_id)[:8]}...")

    # Display conversation history
    if st.session_state.current_conv_id:
        conversation = st.session_state.chat_system.get_conversation(st.session_state.current_conv_id)
        conv_id_str = str(st.session_state.current_conv_id)
        if conversation and conversation.messages:
            # Use original queries for display
            original_queries = st.session_state.original_queries.get(conv_id_str, [])
            formatted_prompts = st.session_state.formatted_prompts.get(conv_id_str, [])
            for i, msg in enumerate(conversation.messages):
                # Use original query if available, otherwise fall back to stored prompt
                user_query = original_queries[i] if i < len(original_queries) else msg.prompt
                formatted_prompt = formatted_prompts[i] if i < len(formatted_prompts) else None
                
                with st.chat_message("user"):
                    st.write(user_query)
                    # Show formatted prompt in expander
                    if formatted_prompt:
                        with st.expander("🔍 View prompt sent to agent"):
                            st.code(formatted_prompt, language=None)
                
                with st.chat_message("assistant"):
                    st.write(msg.response)

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = send_message(prompt)
                    if response:
                        st.write(response)
                    else:
                        st.error("Failed to get response. Check the error message above.")
                        # Don't rerun on error so user can see what went wrong
                        st.stop()
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                    st.stop()

        st.rerun()


if __name__ == "__main__":
    main()
