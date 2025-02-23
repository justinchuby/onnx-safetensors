"""Use safetensors with ONNX."""

from onnx_safetensors import utils
from onnx_safetensors._safetensors_io import (
    apply_tensors,
    load,
    load_file,
    save,
    save_file,
    strip_raw_data,
)

__all__ = [
    "apply_tensors",
    "load",
    "load_file",
    "save",
    "save_file",
    "strip_raw_data",
    "utils",
]
__version__ = "0.1.1"
