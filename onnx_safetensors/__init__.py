"""Use safetensors with ONNX."""

from onnx_safetensors import _utils
from onnx_safetensors._safetensors_io import (
    apply_tensors,
    load,
    load_file,
    save,
    save_file,
    strip_raw_data,
)

__all__ = [
    "_utils",
    "apply_tensors",
    "load",
    "load_file",
    "save",
    "save_file",
    "strip_raw_data",
]
__version__ = "1.0.0"
