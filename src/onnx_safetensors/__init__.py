"""Use safetensors with ONNX."""

__all__ = [
    "load",
    "load_file",
    "load_file_as_external_data",
    "replace_tensors",
    "save",
    "save_file",
]

from onnx_safetensors._safetensors_io import (
    load,
    load_file,
    load_file_as_external_data,
    replace_tensors,
    save,
    save_file,
)

__version__ = "1.2.0"
