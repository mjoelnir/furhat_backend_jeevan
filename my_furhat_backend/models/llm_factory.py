"""
Language Model Factory Module

Design choices:
- Provide a unified interface (BaseLLM) for multiple backends.
- Prefer minimal deps where possible; Ollama can run without torch/transformers.
- HuggingFace/LlamaCpp paths are guarded behind availability checks to avoid hard
  failures when GPU/libtorch is missing.
"""

from abc import ABC, abstractmethod
import multiprocessing
import os
import requests
import shutil
import subprocess
import time

try:
    import torch  # type: ignore[import]
except Exception as e:  # noqa: BLE001
    # Torch is only required for local HuggingFace / Llama models.
    # Ollama-based flows can run without it.
    print(f"[llm_factory] Torch unavailable, GPU-backed HF/llama models disabled: {e}")
    torch = None  # type: ignore[assignment]

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.chat_models import ChatOllama

try:
    from transformers import pipeline  # type: ignore[import]
except Exception as e:  # noqa: BLE001
    print(f"[llm_factory] Transformers pipeline unavailable, HF models disabled: {e}")
    pipeline = None  # type: ignore[assignment]

from my_furhat_backend.config.settings import config
from my_furhat_backend.utils.gpu_utils import (
    setup_gpu,
    move_model_to_device,
    print_gpu_status,
    clear_gpu_cache,
)

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""
    
    @abstractmethod
    def query(self, text: str, tool: bool = False) -> str:
        """
        Process a query with the language model.

        tool flag is for implementations that support tool-calling; default
        implementations may ignore it. Kept simple to avoid overfitting to any
        specific provider API.
        """
        pass

    @abstractmethod
    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Optional tool-binding hook; no-op for backends that don't support it.
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

        Design: prefer running local HF pipelines only when torch+transformers
        are present; otherwise caller should select Ollama.
        """
        self.model_id = model_id
        self.task = task
        # We require both torch and transformers.pipeline to be available
        if torch is None or pipeline is None:
            raise RuntimeError(
                "Torch/Transformers pipeline is not available; HuggingFaceLLM cannot be "
                "initialized. Use an Ollama-backed model instead or install a compatible "
                "PyTorch/Transformers build."
            )

        self.device_info = setup_gpu()

        # Generation defaults (can be overridden by kwargs)
        self.max_length = kwargs.pop("max_length", 1024)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 256)
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.do_sample = kwargs.pop("do_sample", True)
        self.extra_generation_kwargs = {}
        for key in ("min_length", "no_repeat_ngram_size", "repetition_penalty"):
            if key in kwargs:
                self.extra_generation_kwargs[key] = kwargs.pop(key)

        # torch-related model kwargs
        dtype = torch.float16 if self.device_info["cuda_available"] else torch.float32
        self.model_kwargs = {"torch_dtype": dtype}

        # Choose device for the pipeline: CUDA -> 0, CPU -> -1, fallback to MPS string if available
        if self.device_info["cuda_available"]:
            self.pipeline_device = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.pipeline_device = "mps"
        else:
            self.pipeline_device = -1

        # Allow explicit overrides via model_kwargs argument
        extra_model_kwargs = kwargs.pop("model_kwargs", {})
        self.model_kwargs.update(extra_model_kwargs)

        # Create the pipeline with optimized settings
        self.__create_pipeline()
        
    def __del__(self):
        """Cleanup when the model is destroyed."""
        clear_gpu_cache()
    
    def __create_pipeline(self):
        """Create the HuggingFace pipeline with optimized settings; fallback to CPU on failure."""
        try:
            # Clear GPU cache before loading
            clear_gpu_cache()
            
            # Create the pipeline with optimized settings
            self.pipeline = pipeline(
                task=self.task,
                model=self.model_id,
                device=self.pipeline_device,
                model_kwargs=self.model_kwargs
            )
            
            # Move model to GPU if available
            if self.device_info["cuda_available"]:
                self.pipeline = move_model_to_device(self.pipeline, self.device_info["device"])
            
            print_gpu_status()
            
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            # Fallback to CPU if GPU fails
            self.pipeline = pipeline(
                task=self.task,
                model=self.model_id,
                device=-1,
                model_kwargs={"torch_dtype": torch.float32}
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
                
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            return tokenizer.decode(tokens["input_ids"][0])
        except Exception as e:
            print(f"Error truncating input: {e}")
            return prompt
    
    def query(self, prompt: str) -> str:
        """
        Process a query with optimized inference parameters.

        Keeps generation config minimal and truncates inputs to avoid OOM.
        """
        try:
            # Truncate input to prevent OOM errors
            truncated_prompt = self.__truncate_input(prompt)
            
            if self.pipeline:
                # Optimize generation parameters
                generation_config = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.do_sample,
                    "num_beams": 1,
                    "pad_token_id": getattr(
                        self.pipeline.tokenizer, "eos_token_id", None
                    ),
                    "use_cache": True,
                    "return_dict_in_generate": True,
                }
                generation_config.update(self.extra_generation_kwargs)
                
                # Generate response with optimized parameters
                result = self.pipeline(truncated_prompt, **generation_config)
                
                if isinstance(result, list) and len(result) > 0:
                    entry = result[0]
                    if isinstance(entry, dict):
                        if "generated_text" in entry:
                            return entry["generated_text"]
                        if "summary_text" in entry:
                            return entry["summary_text"]
                        # Some pipelines wrap text differently
                        if "text" in entry:
                            return entry["text"]
                    return entry
                return str(result)
            else:
                # Fallback to API if pipeline fails
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{self.model_id}",
                    headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"},
                    json={
                        "inputs": truncated_prompt,
                        "parameters": {
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "do_sample": self.do_sample,
                            **self.extra_generation_kwargs,
                        },
                    },
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
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        system_prompt: str | None = None,
        **kwargs
    ):
        """
        Args:
            model: Ollama model name/tag (e.g., 'llama3.2:latest', 'mixtral:8x7b-instruct', 'qwen2.5:14b-instruct').
            base_url: Ollama server URL.
            **kwargs: Generation/runtime options (temperature, top_p, num_ctx, num_gpu, repeat_penalty, etc.).
        """
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.gen_kwargs = {
            # Reasonable multilingual/chat defaults; override via **kwargs
            "temperature": 0.8,
            "top_p": 0.9,
            "num_ctx": 8192,     # increase if you need longer prompts
            # You can pass num_gpu, num_thread, repeat_penalty, stop, etc.
        }
        self.gen_kwargs.update(kwargs)

        self.chat_llm = None
        self.fallback_llm = None

        if not self._initialize_chat_llm():
            print(
                "[OllamaLLM] Ollama server unavailable. Falling back to HuggingFace model."
            )
            fallback_model = os.getenv(
                "OLLAMA_FALLBACK_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            )
            try:
                self.fallback_llm = HuggingFaceLLM(model_id=fallback_model)
            except Exception as e:
                print(f"[OllamaLLM] Failed to initialize fallback HuggingFace model: {e}")

    _boot_attempted = False
    _model_pull_attempted = False

    def _initialize_chat_llm(self) -> bool:
        if not self._ensure_ollama_server():
            return False
        self._pull_model()
        try:
            self.chat_llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                system=self.system_prompt,
                options=self.gen_kwargs,
            )
            return True
        except Exception as e:
            print(f"[OllamaLLM] Failed to initialize ChatOllama: {e}")
            self.chat_llm = None
            return False

    def _ping_server(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
            return True
        except Exception:
            return False

    def _start_ollama_process(self) -> bool:
        ollama_binary = shutil.which("ollama")
        if not ollama_binary:
            return False

        try:
            subprocess.Popen(
                [ollama_binary, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # give the server a moment to start
            for _ in range(5):
                time.sleep(1)
                if self._ping_server(timeout=1.0):
                    return True
        except Exception as e:
            print(f"[OllamaLLM] Failed to start Ollama process: {e}")
        return False

    def _ensure_ollama_server(self) -> bool:
        if self._ping_server():
            return True

        if not OllamaLLM._boot_attempted:
            OllamaLLM._boot_attempted = True
            if self._start_ollama_process():
                return True

        return self._ping_server(timeout=3.0)

    def _pull_model(self) -> bool:
        if OllamaLLM._model_pull_attempted:
            return False

        ollama_binary = shutil.which("ollama")
        if not ollama_binary:
            print("[OllamaLLM] Cannot pull model because the 'ollama' binary is not in PATH.")
            OllamaLLM._model_pull_attempted = True
            return False

        print(f"[OllamaLLM] Attempting to pull model '{self.model}' via 'ollama pull'.")
        try:
            subprocess.run(
                [ollama_binary, "pull", self.model],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            OllamaLLM._model_pull_attempted = True
            return True
        except Exception as e:
            print(f"[OllamaLLM] Failed to pull model '{self.model}': {e}")
            OllamaLLM._model_pull_attempted = True
            return False

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
        if self.chat_llm is not None:
            try:
                print_gpu_status()
                out = self.chat_llm.invoke(text)
                print_gpu_status()
                return getattr(out, "content", str(out))
            except Exception as e:
                print(f"[OllamaLLM] query error: {e}")
                # Mark Ollama connection unusable so we can transparently
                # fall back to HuggingFace on subsequent calls.
                if "status code 404" in str(e) and self._pull_model():
                    if self._initialize_chat_llm():
                        try:
                            out = self.chat_llm.invoke(text)
                            return getattr(out, "content", str(out))
                        except Exception as second_e:
                            print(f"[OllamaLLM] query error after pulling model: {second_e}")
                self.chat_llm = None
                if self.fallback_llm is None:
                    fallback_model = os.getenv(
                        "OLLAMA_FALLBACK_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct"
                    )
                    try:
                        self.fallback_llm = HuggingFaceLLM(model_id=fallback_model)
                    except Exception as err:
                        print(
                            f"[OllamaLLM] Failed to initialize fallback HuggingFace model: {err}"
                        )

        fallback_response = self._invoke_fallback(text, tool)
        if fallback_response is not None:
            return fallback_response

        return (
            "Ollama backend is unavailable and no fallback model could be initialized."
        )

    def _invoke_fallback(self, text: str, tool: bool) -> str | None:
        if self.fallback_llm is None:
            return None

        try:
            return self.fallback_llm.query(text, tool=tool)
        except TypeError:
            return self.fallback_llm.query(text)

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
