"""Use safetensors with ONNX."""

__all__ = [
    "extract_safetensors_model",
    "load",
    "load_file",
    "load_file_as_external_data",
    "replace_tensors",
    "save",
    "save_file",
    "save_model",
    "save_safetensors_model",
]

from onnx_safetensors._safetensors_io import (
    extract_safetensors_model,
    load,
    load_file,
    load_file_as_external_data,
    replace_tensors,
    save,
    save_file,
    save_model,
    save_safetensors_model,
)

__version__ = "1.5.0"
