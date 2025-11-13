"""
Chatbot Factory Module

This module provides a factory pattern implementation for creating different types of chatbots.
It supports both HuggingFace and LlamaCpp-based chatbots, with a common interface for
conversation handling and response generation.

Key Components:
    - BaseChatbot: Abstract base class defining the chatbot interface
    - Chatbot_HuggingFace: Implementation using HuggingFace models
    - Chatbot_LlamaCpp: Implementation using LlamaCpp models
    - create_chatbot: Factory function for instantiating chatbots
"""

from abc import ABC, abstractmethod
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from my_furhat_backend.utils.util import (
    clean_hc_response,
    format_chatml,
    format_structured_prompt,
    parse_structured_response,
)
from .llm_factory import create_llm

class BaseChatbot(ABC):
    """
    Abstract base class for chatbot implementations.
    
    This class defines the interface that all chatbot implementations must follow,
    ensuring consistent behavior across different model types.
    """
    
    @abstractmethod
    def chatbot(self, state: dict) -> dict:
        """
        Process the conversation state and return an updated state.
        
        Args:
            state (dict): Current conversation state containing messages
            
        Returns:
            dict: Updated conversation state with new AI response
            
        Raises:
            ValueError: If no messages are found in the state
        """
        pass

class Chatbot_HuggingFace(BaseChatbot):
    """
    HuggingFace-based chatbot implementation.
    
    This class provides a concrete implementation of the BaseChatbot interface
    using HuggingFace models for conversation handling.
    """
    
    def __init__(self, model_instance=None, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs):
        """
        Initialize the HuggingFace chatbot.
        
        Args:
            model_instance: Optional pre-initialized language model instance
            model_id (str): HuggingFace model identifier
            **kwargs: Additional model configuration parameters
        """
        if model_instance is not None:
            self.llm = model_instance
        else:
            self.llm = create_llm("huggingface", model_id=model_id, **kwargs)
    
    def chatbot(self, state: dict) -> dict:
        """
        Process conversation state using HuggingFace model.
        
        Args:
            state (dict): Current conversation state
            
        Returns:
            dict: Updated state with new AI response
            
        Raises:
            ValueError: If no messages are found in the state
        """
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        
        prompt = format_chatml(messages)
        response = self.llm.query(prompt)
        
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        
        response_text = clean_hc_response(response_text)
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

class Chatbot_LlamaCpp(BaseChatbot):
    """
    LlamaCpp-based chatbot implementation.
    
    This class provides a concrete implementation of the BaseChatbot interface
    using LlamaCpp models for conversation handling.
    """
    
    def __init__(self, model_instance=None, model_id: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        """
        Initialize the LlamaCpp chatbot.
        
        Args:
            model_instance: Optional pre-initialized language model instance
            model_id (str): Path to the LlamaCpp model file
            **kwargs: Additional model configuration parameters
        """
        if model_instance is not None:
            self.llm = model_instance
        else:
            self.llm = create_llm("llama", model_id=model_id, **kwargs)
        
    def chatbot(self, state: dict) -> dict:
        """
        Process conversation state using LlamaCpp model.
        
        Args:
            state (dict): Current conversation state
            
        Returns:
            dict: Updated state with new AI response
            
        Raises:
            ValueError: If no messages are found in the state
        """
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        
        prompt = format_structured_prompt(messages)
        response = self.llm.query(prompt)
        
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        
        response_text = parse_structured_response(response_text)
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

class Chatbot_Ollama(BaseChatbot):
    """
    Ollama-based chatbot implementation, with optional preferred_language.
    Uses the Ollama backend added to llm_factory (create_llm('ollama', ...)).
    """
    def __init__(
        self,
        model_instance=None,
        model: str = "llama3.1:instruct",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        if model_instance is not None:
            self.llm = model_instance
        else:
            self.llm = create_llm("ollama", model=model, base_url=base_url, **kwargs)

    # (Optional) allow external control at runtime
    def set_language(self, lang: str | None):
        self._forced_language = lang  # mutable, optional override

    def _infer_language(self, text: str) -> str | None:
        """
        Super-light heuristic: detect Norwegian vs English vs fallback.
        Swap to a real detector if you like.
        """
        if not text:
            return None
        t = text.lower()
        # quick Norwegian markers
        no_markers = ("å", "ø", "æ", "ikke", "også", "hvordan", "hva", "hvorfor", "forklar", "beskriv")
        if any(m in t for m in no_markers):
            return "Norwegian"
        # quick French/Spanish markers (examples)
        fr_markers = ("pourquoi", "comment", "était", "être", "œ")
        es_markers = ("por qué", "cómo", "está", "ser")
        if any(m in t for m in fr_markers): return "French"
        if any(m in t for m in es_markers): return "Spanish"
        # default: None -> let model pick, or your app can set state["preferred_language"]
        return None

    def chatbot(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")

        # 1) Runtime language decision (priority: explicit state -> forced -> inferred)
        lang = state.get("preferred_language", None)
        if lang is None:
            lang = getattr(self, "_forced_language", None)

        if lang is None:
            # infer from last HumanMessage
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    lang = self._infer_language(m.content)
                    break

        # 2) Inject a per-turn SystemMessage if we have a language
        if lang:
            already_has_lang = any(
                isinstance(m, SystemMessage) and "respond in" in (m.content or "").lower()
                for m in messages
            )
            if not already_has_lang:
                messages = [SystemMessage(
                    content=(
                        f"Please respond in {lang}. "
                        "If structured output is required (e.g., JSON), DO NOT translate keys."
                    )
                )] + messages

        # 3) Format & query as usual
        prompt = format_chatml(messages)
        response = self.llm.query(prompt)

        # 4) Normalize to text and append
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)

        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        # Optionally persist the decided language back into state for next turn
        state["preferred_language"] = lang or state.get("preferred_language")
        return state

def create_chatbot(chatbot_type: str, **kwargs) -> BaseChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        chatbot_type (str): Type of chatbot to create ("huggingface" or "llama")
        **kwargs: Additional initialization parameters
        
    Returns:
        BaseChatbot: Instance of the specified chatbot type
        
    Raises:
        ValueError: If an unsupported chatbot type is provided
    """
    if chatbot_type.lower() == "huggingface":
        return Chatbot_HuggingFace(**kwargs)
    elif chatbot_type.lower() == "llama":
        return Chatbot_LlamaCpp(**kwargs)
    elif chatbot_type.lower() == "ollama":
        return Chatbot_Ollama(**kwargs)
    else:
        raise ValueError(f"Unsupported chatbot type: {chatbot_type}")

__all__ = ["create_chatbot", "Chatbot_HuggingFace", "Chatbot_LlamaCpp", "Chatbot_Ollama", "BaseChatbot"]
