"""
Language Model Factory Module

This module provides a factory pattern implementation for creating and managing different types of language models.
It supports both HuggingFace and LlamaCpp models with GPU optimization and monitoring capabilities.

Classes:
    BaseLLM: Abstract base class defining the interface for all LLM implementations.
    HuggingFaceLLM: Implementation using HuggingFace's API and models.
    LlamaCcpLLM: Implementation using LlamaCpp for local model inference.

Functions:
    create_llm: Factory function to create instances of different LLM types.
"""

from abc import ABC, abstractmethod
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.chat_models import ChatOllama
from my_furhat_backend.config.settings import config
from my_furhat_backend.utils.gpu_utils import setup_gpu, move_model_to_device, print_gpu_status, clear_gpu_cache
from transformers import pipeline
import torch
import os
import requests

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""
    
    @abstractmethod
    def query(self, text: str, tool: bool = False) -> str:
        """
        Process a query with the language model.
        
        Args:
            text (str): The input text or prompt to be processed.
            tool (bool): If True, invoke the model with pre-bound tools.
            
        Returns:
            str: The generated response from the language model.
        """
        pass

    @abstractmethod
    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind external tools to the language model for extended functionality.
        
        Args:
            tools (list): A list of tools to be bound to the language model.
            tool_schema (dict | str, optional): The schema or configuration for the tools.
        """
        pass

class HuggingFaceLLM(BaseLLM):
    """
    Implementation of BaseLLM using HuggingFace's API and models.
    
    This class provides functionality to interact with HuggingFace models either through
    their API or local pipeline, with GPU optimization and monitoring capabilities.
    """
    
    def __init__(self, model_id: str, task: str = "text-generation", **kwargs):
        """
        Initialize the HuggingFace LLM with optimized settings.
        
        Args:
            model_id (str): The model identifier from HuggingFace
            task (str): The task type (default: "text-generation")
            **kwargs: Additional arguments for model configuration
        """
        self.model_id = model_id
        self.task = task
        self.device_info = setup_gpu()
        
        # Optimize model loading
        self.model_kwargs = {
            "device_map": "auto",  # Automatically handle device placement
            "torch_dtype": torch.float16,  # Use half precision
            "low_cpu_mem_usage": True,  # Optimize CPU memory usage
            "load_in_8bit": True,  # Use 8-bit quantization
            "max_memory": {0: "16GB"} if self.device_info["cuda_available"] else None
        }
        
        # Update with any additional kwargs
        self.model_kwargs.update(kwargs)
        
        # Create the pipeline with optimized settings
        self.__create_pipeline()
        
    def __del__(self):
        """Cleanup when the model is destroyed."""
        clear_gpu_cache()
    
    def __create_pipeline(self):
        """Create the HuggingFace pipeline with optimized settings."""
        try:
            # Clear GPU cache before loading
            clear_gpu_cache()
            
            # Create the pipeline with optimized settings
            self.pipeline = pipeline(
                task=self.task,
                model=self.model_id,
                **self.model_kwargs
            )
            
            # Move model to GPU if available
            if self.device_info["cuda_available"]:
                self.pipeline = move_model_to_device(self.pipeline, self.device_info["device"])
            
            print_gpu_status()
            
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            # Fallback to CPU if GPU fails
            self.model_kwargs["device_map"] = "cpu"
            self.pipeline = pipeline(
                task=self.task,
                model=self.model_id,
                **self.model_kwargs
            )
    
    def __truncate_input(self, prompt: str) -> str:
        """
        Truncate the input prompt if it exceeds the maximum token length.
        
        Args:
            prompt (str): The input prompt.
            
        Returns:
            str: The truncated prompt.
        """
        try:
            tokenizer = self.pipeline.tokenizer if self.pipeline else None
            if not tokenizer:
                return prompt
                
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model_kwargs["max_length"])
            return tokenizer.decode(tokens["input_ids"][0])
        except Exception as e:
            print(f"Error truncating input: {e}")
            return prompt
    
    def query(self, prompt: str) -> str:
        """
        Process a query with optimized inference parameters.
        
        Args:
            prompt (str): The input text or prompt to be processed.
            
        Returns:
            str: The generated response from the language model.
        """
        try:
            # Truncate input to prevent OOM errors
            truncated_prompt = self.__truncate_input(prompt)
            
            if self.pipeline:
                # Optimize generation parameters
                generation_config = {
                    "max_new_tokens": 256,  # Reduced from 512
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "num_beams": 1,  # Use greedy decoding for speed
                    "pad_token_id": self.pipeline.tokenizer.eos_token_id,
                    "use_cache": True,  # Enable KV cache
                    "return_dict_in_generate": True
                }
                
                # Generate response with optimized parameters
                result = self.pipeline(
                    truncated_prompt,
                    **generation_config
                )
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0]["generated_text"]
                    return result[0]
                return str(result)
            else:
                # Fallback to API if pipeline fails
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{self.model_id}",
                    headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"},
                    json={
                        "inputs": truncated_prompt,
                        "parameters": {
                            "max_new_tokens": 256,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "do_sample": True
                        }
                    }
                )
                response.raise_for_status()
                return response.json()[0]["generated_text"]
                
        except Exception as e:
            print(f"Error in query: {e}")
            return f"Error processing query: {str(e)}"

    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind tools to the LLM for enhanced functionality.
        
        Args:
            tools (list): List of tools to bind.
            tool_schema (dict | str, optional): Schema for the tools.
        """
        pass

class LlamaCcpLLM(BaseLLM):
    """
    Implementation of BaseLLM using LlamaCpp for local model inference.
    
    This class provides functionality to interact with LlamaCpp models locally,
    with GPU optimization and monitoring capabilities.
    """
    
    def __init__(self, model_id: str = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        """
        Initialize the LlamaCcpLLM.
        
        Args:
            model_id (str): Name of the GGUF model file
            **kwargs: Additional generation parameters.
        """
        super().__init__()
        
        device_info = setup_gpu()
        print_gpu_status()
        
        # Get the full path to the GGUF model
        model_path = os.path.join(config["GGUF_MODELS_PATH"], model_id)
        
        # Verify model file exists and is accessible
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found at {model_path}")
            
        # Check file size to ensure it's not empty or corrupted
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Arbitrary minimum size for a GGUF model
            raise ValueError(f"Model file at {model_path} appears to be corrupted or incomplete (size: {file_size} bytes)")
            
        # Check file permissions
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"No read permission for model file at {model_path}")
            
        print(f"Loading model from: {model_path}")
        print(f"Model file size: {file_size / (1024*1024):.2f} MB")
        
        # Set default parameters
        kwargs.setdefault("n_ctx", 10000)
        kwargs.setdefault("n_gpu_layers", 32)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("n_batch", 512)
        kwargs.setdefault("max_tokens", 1024)
        kwargs.setdefault("repeat_penalty", 1.5)
        kwargs.setdefault("top_p", 0.5)
        kwargs.setdefault("verbose", True)
        
        # Set n_threads only if not already provided in kwargs
        if "n_threads" not in kwargs:
            kwargs["n_threads"] = multiprocessing.cpu_count() - 1
        
        try:
            self.chat_llm = ChatLlamaCpp(
                model_path=model_path,
                do_sample=True,
                **kwargs
            )
            
            if device_info["cuda_available"]:
                self.chat_llm = move_model_to_device(self.chat_llm, device_info["device"])
            
            print_gpu_status()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Model path: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")
            print(f"Model file permissions: {oct(os.stat(model_path).st_mode)[-3:] if os.path.exists(model_path) else 'N/A'}")
            raise
    
    def __del__(self):
        """Cleanup when the model is destroyed."""
        clear_gpu_cache()

    def query(self, text: str, tool: bool = False) -> str:
        """
        Process the query using the LlamaCpp model.
        
        Args:
            text (str): The input query or prompt.
            tool (bool): If True, use the tool-bound version of the model.
            
        Returns:
            str: The generated response from the chat model.
        """
        print_gpu_status()
        
        try:
            clear_gpu_cache()
            
            if tool:
                response = self.chat_llm_with_tools.invoke(text)
            else:
                response = self.chat_llm.invoke(text)
            
            print_gpu_status()
            return response
        except Exception as e:
            print(f"Error in query: {e}")
            return ""

    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind external tools to the LlamaCcpLLM.
        
        Args:
            tools (list): List of tools to bind.
            tool_schema (dict | str, optional): Schema for the tools.
        """
        self.chat_llm.bind_tools(tools)

class OllamaLLM(BaseLLM):
    """
    LLM implementation using an Ollama server (http://localhost:11434 by default).
    Suitable for multilingual models like llama3.1, mixtral, qwen2.5, etc.
    Mirrors the query/bind_tools interface of other backends.
    """

    def __init__(
        self,
        model: str = "llama3.1:instruct",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Args:
            model: Ollama model name/tag (e.g., 'llama3.1:instruct', 'mixtral:8x7b-instruct', 'qwen2.5:14b-instruct').
            base_url: Ollama server URL.
            **kwargs: Generation/runtime options (temperature, top_p, num_ctx, num_gpu, repeat_penalty, etc.).
        """
        self.model = model
        self.base_url = base_url
        self.gen_kwargs = {
            # Reasonable multilingual/chat defaults; override via **kwargs
            "temperature": 0.8,
            "top_p": 0.9,
            "num_ctx": 8192,     # increase if you need longer prompts
            # You can pass num_gpu, num_thread, repeat_penalty, stop, etc.
        }
        self.gen_kwargs.update(kwargs)

        # Eager check that Ollama server is reachable (optional but helpful)
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
        except Exception as e:
            print(f"[OllamaLLM] Warning: Could not reach Ollama at {self.base_url}: {e}")

        # LangChain wrapper; keeps your interface consistent with ChatLlamaCpp
        # ChatOllama accepts model/base_url and a dict of 'options' for runtime
        self.chat_llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            # map gen kwargs into options LangChain forwards to Ollama
            options=self.gen_kwargs
        )

    def __del__(self):
        try:
            clear_gpu_cache()
        except Exception:
            pass

    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind tool specs to this chat model (LangChain will convert tools into
        the proper JSON schema format for tool calling).
        """
        try:
            self.chat_llm = self.chat_llm.bind_tools(tools)
        except Exception as e:
            print(f"[OllamaLLM] bind_tools error: {e}")

    def query(self, text: str, tool: bool = False) -> str:
        """
        Send a prompt to the Ollama model. If you've bound tools, LangChain will handle tool calling.
        """
        try:
            print_gpu_status()
            # For Chat models, we can use .invoke with a simple human message
            # If you prefer plain text, LangChain accepts string directly.
            out = self.chat_llm.invoke(text)
            print_gpu_status()
            # ChatOllama returns a BaseMessage or string depending on version;
            # extract text robustly:
            return getattr(out, "content", str(out))
        except Exception as e:
            print(f"[OllamaLLM] query error: {e}")
            return ""

def create_llm(llm_type: str, **kwargs) -> BaseLLM:
    """
    Factory function to create an instance of a language model.
    
    Args:
        llm_type (str): Type of LLM to create ("huggingface" or "llama").
        **kwargs: Additional parameters for the LLM constructor.
        
    Returns:
        BaseLLM: An instance of the specified LLM type.
        
    Raises:
        ValueError: If the specified llm_type is not supported.
    """
    if llm_type == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif llm_type == "llama":
        return LlamaCcpLLM(**kwargs)
    elif llm_type == "ollama":
        return OllamaLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

__all__ = ["create_llm", "HuggingFaceLLM", "LlamaCcpLLM", "OllamaLLM", "BaseLLM"]
