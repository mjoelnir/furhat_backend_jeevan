"""
GPU Utilities Module

Design choice: keep this lightweight and resilient.
- Prefer PyTorch when available; otherwise fall back to CPU/psutil so the backend
  can still run in Ollama-only mode without hard failures.
- Avoid extra dependencies beyond psutil (for RAM stats) to keep the footprint small.

Functions:
    setup_gpu: Detect CUDA presence and report device/memory info.
    move_model_to_device: Safely move a model if torch is available.
    print_gpu_status: Debug helper for quick memory snapshots.
    clear_gpu_cache: Torch empty_cache wrapper; no-op if CUDA/torch missing.
"""

from typing import Optional, Dict, Any

import psutil

try:
    import torch  # type: ignore[import]
except Exception as e:  # noqa: BLE001
    print(f"[gpu_utils] Torch unavailable, using CPU-only utilities: {e}")
    torch = None  # type: ignore[assignment]

def setup_gpu() -> Dict[str, Any]:
    """
    Detect CUDA and gather basic device/memory info.

    Chosen over heavier GPU libs to minimize dependencies; psutil covers CPU stats,
    torch (if present) covers CUDA stats. Returns a dict so callers can log/branch
    without importing torch themselves.
    """
    device_info: Dict[str, Any] = {
        "cuda_available": False,
        "device": None,
        "device_name": None,
        "memory_info": None,
    }

    if torch is not None and torch.cuda.is_available():
        device_info["cuda_available"] = True
        device_info["device"] = torch.device("cuda")
        device_info["device_name"] = torch.cuda.get_device_name()
        device_info["memory_info"] = {
            "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "cached": torch.cuda.memory_reserved() / 1024**2,  # MB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
        }
    else:
        device_info["device"] = None
        device_info["device_name"] = "CPU"
        device_info["memory_info"] = {
            "total": psutil.virtual_memory().total / 1024**2,  # MB
            "available": psutil.virtual_memory().available / 1024**2,  # MB
            "used": psutil.virtual_memory().used / 1024**2,  # MB
        }

    return device_info

def move_model_to_device(model: Any, device: Optional["torch.device"] = None) -> Any:  # type: ignore[name-defined]
    """
    Move a model to a target device if torch is present.

    Why this approach: keeps callers simple and safe—if torch is absent or the
    model lacks `.to()`, we return the model unchanged instead of failing.
    """
    if torch is None:
        # No-op if torch is unavailable
        return model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "to"):
        return model.to(device)
    return model

def print_gpu_status() -> None:
    """
    Print a quick memory snapshot for debugging (GPU if available, else CPU).

    Chosen to avoid extra tooling; uses torch/psutil only, so it works on both
    CUDA and CPU-only setups without additional installs.
    """
    if torch is not None and torch.cuda.is_available():
        print("\n=== GPU Status ===")
        print(f"CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("\n=== CPU Status ===")
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / 1024**2:.2f} MB")
        print(f"Available Memory: {memory.available / 1024**2:.2f} MB")
        print(f"Used Memory: {memory.used / 1024**2:.2f} MB")

def clear_gpu_cache() -> None:
    """
    Clear unused GPU cache via torch.cuda.empty_cache (no-op if torch/CUDA missing).

    Kept minimal to avoid surprises on CPU-only deployments; chosen over custom
    allocators because torch already manages its own caching.
    """
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")