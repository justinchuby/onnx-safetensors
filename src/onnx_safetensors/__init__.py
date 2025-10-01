"""Use safetensors with ONNX."""

__all__ = [
    "apply_tensors",
    "load",
    "load_file",
    "load_file_as_external_data",
    "read_safetensors",
    "replace_tensors",
    "save",
    "save_file",
]

from onnx_safetensors._safetensors_io import (
    apply_tensors,
    load,
    load_file,
    load_file_as_external_data,
    read_safetensors,
    replace_tensors,
    save,
    save_file,
)

__version__ = "1.2.0"
