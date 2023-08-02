"""Use safetensors with ONNX."""

from onnx_safetensors._safetensors_io import (
    load,
    load_file,
    save,
    save_file,
    strip_raw_data,
)

__all__ = [
    "load",
    "load_file",
    "save",
    "save_file",
    "strip_raw_data",
]
__version__ = "0.1.0"
