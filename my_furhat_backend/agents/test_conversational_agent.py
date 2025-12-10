"""
Example conversational agent using LangGraph (not used in production).

Design: kept as a reference scaffold to show a LangGraph flow with RAG and
tool-style context injections. Uses a legacy RAG signature and static model
path; update or remove for production.
"""

import os
import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
# Import uuid to generate unique tool call IDs.

# Import your custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace, Chatbot_LlamaCpp, create_chatbot
from my_furhat_backend.utils.util import clean_output

# Import your RAG class.
from my_furhat_backend.RAG.rag_flow import RAG

# --- Global Initialization ---

# Create a global instance of RAG with document ingestion parameters.
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/ingestion/CMRPublished.pdf"
)

# Define the global chatbot instance.
# Uncomment the following line to use the HuggingFace-based chatbot.
# chatbot = create_chatbot("huggingface")
# Here we use a Llama-based chatbot with a specific model path.
chatbot = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")

# --- Define Conversation State ---
class State(TypedDict):
    """
    Typed dictionary representing the conversation state.
    
    Attributes:
        messages (List[BaseMessage]): The conversation history as a list of messages.
        input (str): The current user input.
    """
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

# --- Graph Node Functions ---
# These functions define the nodes in the conversation workflow graph.

def input_node(state: State) -> dict:
    """
    Capture the user's query and add it to the chat history.

    This node extracts the "input" field from the state, wraps it in a HumanMessage,
    and appends it to the state's messages list.

    Parameters:
        state (State): The current conversation state.

    Returns:
        dict: A dictionary containing the updated messages list.
    """
    # Ensure the messages list exists.
    state.setdefault("messages", [])
    # Create a HumanMessage from the user input.
    human_msg = HumanMessage(content=state.get("input", ""))
    state["messages"].append(human_msg)
    return {"messages": state["messages"]}


def retrieval_node(state: State) -> dict:
    """
    Retrieve relevant document context using the RAG system.

    Uses the global RAG instance to find and retrieve document chunks relevant to the query.
    The retrieved content is concatenated and added as a ToolMessage to the conversation history.

    Parameters:
        state (State): The current conversation state.

    Returns:
        dict: A dictionary containing the updated messages list.
    """
    # Extract the user query.
    query = state.get("input", "")
    # Retrieve similar documents using RAG, with reranking enabled.
    retrieved_docs = rag_instance.retrieve_similar(query, rerank=True)
    # Concatenate document page contents if any results are found.
    if retrieved_docs:
        retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
    else:
        retrieved_text = "No relevant document context found."
    # Create a ToolMessage with the retrieved context.
    tool_msg = ToolMessage(
        content=f"Retrieved context:\n{retrieved_text}",
        name="document_retriever",
        tool_call_id=str(uuid.uuid4())
    )
    state["messages"].append(tool_msg)
    return {"messages": state["messages"]}


def uncertainty_check_node(state: State) -> dict:
    """
    Check if the retrieved document context is sufficient.

    This node examines the ToolMessage (if any) to determine whether the retrieved context
    is adequate. If the context is too short (less than 50 characters), it flags uncertainty.

    Parameters:
        state (State): The current conversation state.

    Returns:
        dict: A dictionary with the updated messages list and an 'uncertainty' flag.
    """
    # Find the first ToolMessage in the messages.
    tool_msg = next((msg for msg in state["messages"] if isinstance(msg, ToolMessage)), None)
    # Set uncertainty to True if no ToolMessage is found or if the message content is too short.
    state["uncertainty"] = (tool_msg is None) or (len(tool_msg.content) < 50)
    return {"messages": state["messages"], "uncertainty": state["uncertainty"]}


def uncertainty_response_node(state: State) -> dict:
    """
    Provide a response when insufficient document context is retrieved.

    If the uncertainty flag is set, this node appends an AIMessage asking for clarification.

    Parameters:
        state (State): The current conversation state.

    Returns:
        dict: A dictionary containing the updated messages list.
    """
    # Create an AIMessage to ask for clarification.
    uncertain_msg = AIMessage(content="I'm not certain I have enough context from the document. Could you please clarify your question?")
    state["messages"].append(uncertain_msg)
    return {"messages": state["messages"]}


def generation_node(state: State) -> dict:
    """
    Generate an answer using the chatbot based on the full conversation history.

    This node invokes the chatbot to generate an AI response using the accumulated messages,
    including the initial user input, system prompt, retrieved context, and any uncertainty response.

    Parameters:
        state (State): The current conversation state.

    Returns:
        dict: The conversation state updated with the chatbot's AI response.
    """
    # Use the chatbot to generate the answer.
    return chatbot.chatbot(state)

# --- Build the Graph Workflow ---
# Create a state graph and add nodes and edges to define the conversation flow.
graph = StateGraph(State)
graph.add_node("capture_input", input_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("uncertainty_check", uncertainty_check_node)
graph.add_node("uncertainty_response", uncertainty_response_node)
graph.add_node("generation", generation_node)

# Define the linear progression of nodes from start through retrieval.
graph.add_edge(START, "capture_input")
graph.add_edge("capture_input", "retrieval")
graph.add_edge("retrieval", "uncertainty_check")

# Conditional branch: if uncertainty is detected, route to uncertainty_response; otherwise, proceed to generation.
def condition_func(state):
    """
    Conditional function to determine the next node based on uncertainty.

    Parameters:
        state (State): The current conversation state.

    Returns:
        str: The name of the next node ("uncertainty_response" or "generation").
    """
    return "uncertainty_response" if state.get("uncertainty") else "generation"

graph.add_conditional_edges(
    "uncertainty_check",
    condition_func,
    {"uncertainty_response": "uncertainty_response", "generation": "generation"}
)
# Both conditional branches lead to the end of the conversation.
graph.add_edge("uncertainty_response", END)
graph.add_edge("generation", END)

# Compile the graph with memory checkpointing.
memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)

# --- Run the Conversation ---
def run_conversation_streaming():
    """
    Run the interactive conversation with the document-based assistant.

    This function sets up an initial state with a system prompt, processes the conversation
    through the compiled state graph, and prints the final AI response. It then enters a loop
    to continue the conversation until the user exits.
    """
    # Configuration for the graph execution (e.g., thread ID).
    config = {"configurable": {"thread_id": "1"}}
    print("Welcome. This assistant will discuss only the ingested document.")
    
    # Prompt user for initial query.
    query = input("Enter your query about the document (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        return
    
    # Set up the initial state with a system prompt guiding the conversation.
    system_prompt = SystemMessage(content=(
        "You are a friendly and knowledgeable assistant. "
        "When answering, speak naturally and conversationally—as if you're chatting with a friend. "
        "Use simple, clear language and explain things in a warm, approachable manner. "
        "Answer only based on the provided document content."
    ))
    state: State = {
        "input": query,
        "messages": [HumanMessage(content=system_prompt.content)]
    }
    
    print("Streaming response...")
    # Process the state through the graph; the state is updated in each step.
    for step in compiled_graph.stream(state, config, stream_mode="values"):
        pass
    
    # Retrieve and print only the final AI response from the conversation history.
    final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    if final_ai_msg:
        print("Assistant:", clean_output(final_ai_msg.content))
    else:
        print("No AI response found.")

    # Continue the conversation in a loop.
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        # Update the state with new user input.
        state["input"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        # Process the updated state through the graph.
        for step in compiled_graph.stream(state, config, stream_mode="values"):
            pass
        # Print the latest AI response.
        final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
        if final_ai_msg:
            print("Assistant:", clean_output(final_ai_msg.content))
        else:
            print("No AI response found.")


if __name__ == "__main__":
    run_conversation_streaming()
