"""
Secondary LangGraph conversational agent scaffold (not used in production).

Design: reference/example showing router + RAG + grading flows. Uses legacy
paths and prompts; kept for experimentation only.
"""

import os
import time
import logging
import uuid
import re
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.exceptions import OutputParserException

# Import your custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace, Chatbot_LlamaCpp, create_chatbot
from my_furhat_backend.utils.util import clean_output, extract_json
from my_furhat_backend.config.settings import config

# Import your RAG class.
from my_furhat_backend.RAG.rag_flow import RAG

# ---------------------------
# Define external chains with updated prompts:
# ---------------------------
# Create a global LLM instance from your chatbot; here we use a Llama model.
llm = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf").llm.chat_llm

# Router chain: routes a question to either web search or vectorstore retrieval.
router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
Otherwise, use web-search.
Return only a valid JSON object with a single key "datasource" whose value is either "web_search" or "vectorstore".
Do not include any additional text.
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
# Create a chain that uses the router prompt, then the LLM, then parses the output as JSON.
question_router = router_prompt | llm | JsonOutputParser()

# Generate chain: uses retrieved context to generate a concise answer.
generate_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {document}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
# Helper function to format retrieved documents.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Create a chain for generating an answer.
rag_chain = generate_prompt | llm | StrOutputParser()

# Retrieval Grader chain: assesses the relevance of a retrieved document.
retrieval_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing the relevance of a retrieved document to a user question.
Return only a valid JSON object with a single key "score" whose value is either "yes" or "no".
Do not include any additional text.
Document: {document}
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

# Hallucination Grader chain: checks whether an answer is grounded in the provided facts.
hallucination_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer is grounded in the provided facts.
Return only a valid JSON object with a single key "score" whose value is either "yes" or "no".
Do not include any additional text.
Facts: {documents}
Answer: {generation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

# Answer Grader chain: checks if an answer is useful in resolving a question.
answer_grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an answer is useful to resolve a question.
Return only a valid JSON object with a single key "score" whose value is either "yes" or "no".
Do not include any additional text.
Answer: {generation}
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()

# ---------------------------
# Set up our global instances:
# ---------------------------
# Create a global instance of RAG for document retrieval.
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/ingestion/CMRPublished.pdf"
)
# Create an instance of a web search tool.
search_tool = TavilySearchResults()
# Create a chatbot instance using a Llama model.
chatbot = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")

# ---------------------------
# Define the State.
# ---------------------------
class State(TypedDict):
    """
    Typed dictionary representing the conversation state.

    Attributes:
        messages (List[BaseMessage]): The conversation history.
        question (str): The current question from the user.
        documents (list): Retrieved documents (from RAG or web search).
        web_search (str): Flag ("Yes" or "No") indicating if web search is used.
        generation (str): The generated answer.
        websearch_docs (str): Web search results (if applicable).
    """
    messages: Annotated[List[BaseMessage], "add_messages"]
    question: str
    documents: list
    web_search: str
    generation: str
    websearch_docs: str

# ---------------------------
# Define Workflow Node Functions.
# ---------------------------
def capture_input(state: State) -> dict:
    """
    Capture user input and add it as a HumanMessage to the state.

    Parameters:
        state (State): The conversation state.

    Returns:
        dict: Updated state with the 'messages' list.
    """
    state.setdefault("messages", [])
    human_msg = HumanMessage(content=state["question"])
    state["messages"].append(human_msg)
    return {"messages": state["messages"]}

def route_question(state: State) -> str:
    """
    Route the user question to either web search or vectorstore (RAG) retrieval.

    Uses the router chain to decide the data source.
    If the router outputs "web_search", returns "websearch"; otherwise returns "vectorstore".

    Parameters:
        state (State): The conversation state.

    Returns:
        str: Routing decision ("websearch" or "vectorstore").
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    try:
        source = question_router.invoke({"question": question})
    except Exception as e:
        print("Router chain error:", e)
        source = {"datasource": "vectorstore"}
    print("Router output:", source)
    if source.get('datasource', 'vectorstore') == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    else:
        print("---ROUTE QUESTION TO RAG (vectorstore)---")
        return "vectorstore"

def retrieve(state: State) -> dict:
    """
    Retrieve relevant documents using the RAG system based on the user question.

    Parameters:
        state (State): The conversation state.

    Returns:
        dict: Updated state with retrieved 'documents'.
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = rag_instance.retrieve_similar(question, rerank=True)
    state["documents"] = documents
    return {"documents": documents, "question": question}

def grade_documents(state: State) -> dict:
    """
    Grade retrieved documents for relevance to the question.

    Uses the retrieval grader to filter out irrelevant documents.
    If any document is graded as irrelevant, sets the 'web_search' flag to "Yes".

    Parameters:
        state (State): The conversation state.

    Returns:
        dict: Updated state with filtered 'documents' and 'web_search' flag.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        try:
            raw_output = retrieval_grader.invoke({"question": question, "document": d.page_content})
            json_str = extract_json(raw_output)
            score = json.loads(json_str)
            grade = score.get('score', 'yes')
        except Exception as e:
            print("Retrieval grader error:", e)
            grade = "yes"
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    state["documents"] = filtered_docs
    state["web_search"] = web_search
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def decide_to_generate(state: State) -> str:
    """
    Decide whether to proceed with generation or use web search results.

    If the 'web_search' flag is "Yes", returns "websearch". Otherwise, returns "generate".

    Parameters:
        state (State): The conversation state.

    Returns:
        str: Decision ("websearch" or "generate").
    """
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def web_search(state: State) -> dict:
    """
    Perform a web search using an external search tool and add results to the documents.

    Parameters:
        state (State): The conversation state.

    Returns:
        dict: Updated state with web search results added to 'documents'.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = search_tool.invoke({"query": question})
    # Concatenate the content of all web search results.
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results_doc)
    else:
        documents = [web_results_doc]
    state["documents"] = documents
    return {"documents": documents, "question": question}

def generate(state: State) -> dict:
    """
    Generate an answer using the RAG chain based on the question and retrieved context.

    Parameters:
        state (State): The conversation state.

    Returns:
        dict: Updated state with the generated answer stored in 'generation'.
    """
    print("---GENERATE---")
    question = state["question"]
    context = format_docs(state["documents"])
    generation = rag_chain.invoke({"question": question, "document": context})
    state["generation"] = generation
    return {"documents": state["documents"], "question": question, "generation": generation}

def grade_generation_v_documents_and_question(state: State) -> str:
    """
    Grade the generated answer against the retrieved documents and question.

    First, uses the hallucination grader to assess whether the generation is grounded.
    Then, uses the answer grader to determine if the generation addresses the question.
    
    Returns:
        str: "useful" if the answer is both grounded and addresses the question,
             "not useful" if not addressing the question,
             "not supported" if not grounded (indicating a re-try may be needed).
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    context = format_docs(state["documents"])
    generation = state["generation"]
    try:
        raw_output = hallucination_grader.invoke({"documents": context, "generation": generation})
        json_str = extract_json(raw_output)
        score = json.loads(json_str)
        grade = score.get('score', 'no')
    except Exception as e:
        print("Hallucination grader error:", e)
        grade = "no"
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED---")
        try:
            raw_output = answer_grader.invoke({"question": question, "generation": generation})
            json_str = extract_json(raw_output)
            score = json.loads(json_str)
            grade_ans = score.get('score', 'no')
        except Exception as e:
            print("Answer grader error:", e)
            grade_ans = "no"
        if grade_ans == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED, RE-TRY---")
        return "not supported"

# ---------------------------
# Build the Graph Workflow.
# ---------------------------
workflow = StateGraph(State)

# Add nodes to the workflow graph.
workflow.add_node("capture_input", capture_input)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("decide_to_generate", decide_to_generate)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation_v_documents_and_question)

# Set the initial conditional entry point based on routing decision.
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

# Define the main flow of nodes.
workflow.add_edge("capture_input", "route_question")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile the workflow graph with checkpointing memory.
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)

# ---------------------------
# Run the Conversation.
# ---------------------------
def run_conversation_streaming():
    """
    Run the interactive conversation with the assistant.

    This function sets up the initial conversation state, processes the state through the compiled
    workflow graph, and streams responses from the assistant. It then enters a loop to continue the
    conversation until the user exits.
    """
    config = {"configurable": {"thread_id": "1"}}
    print("Welcome. This assistant uses RAG, web search, and grading routing.")
    query = input("Enter your query about the document (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        return
    # Set a system prompt to guide the conversation.
    system_prompt = SystemMessage(content=(
        "You are a friendly and knowledgeable assistant. "
        "Answer based on the provided document content and web search results if needed."
    ))
    # Initialize the conversation state.
    state: State = {
        "question": query,
        "messages": [HumanMessage(content=system_prompt.content)],
        "documents": [],
        "web_search": "No",
        "generation": "",
        "websearch_docs": ""
    }
    print("Streaming response...")
    # Process the state through the compiled graph (streaming mode).
    for step in compiled_graph.stream(state, config, stream_mode="values"):
        pass
    final_generation = state.get("generation", "")
    if final_generation:
        print("Assistant:", clean_output(final_generation))
    else:
        print("No AI response found.")
    # Continue conversation loop.
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state["question"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        for step in compiled_graph.stream(state, config, stream_mode="values"):
            pass
        final_generation = state.get("generation", "")
        if final_generation:
            print("Assistant:", clean_output(final_generation))
        else:
            print("No AI response found.")

if __name__ == "__main__":
    run_conversation_streaming()
