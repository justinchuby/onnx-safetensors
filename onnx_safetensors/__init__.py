"""Use safetensors with ONNX."""

from ._safetensors_io import (
    load_safetensors_file,
    load_safetensors,
    save_safetensors_file,
)

__all__ = [
    "load_safetensors_file",
    "load_safetensors",
    "save_safetensors_file",
]
__version__ = "0.1.0"
