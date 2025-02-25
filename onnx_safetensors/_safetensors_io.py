"""Private module for loading and saving safetensors data to ONNX models."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Union

import onnx
import onnx.helper
import safetensors
import safetensors.numpy

from onnxscript import ir

if TYPE_CHECKING:
    import numpy as np

ModelOrGraph = Union[
    onnx.ModelProto,
    onnx.GraphProto,
    ir.Model,
    ir.Graph,
]


def load_file(proto: ModelOrGraph, tensor_file: str | os.PathLike) -> set[str]:
    """Load external data into ONNX model from a safetensors file.

    Args:
        proto: ONNX model or graph to load external data into.
        tensor_file: safetensors file to load into ONNX model.

    Returns:
        Names of tensors that were applied.
    """
    applied = set()
    with safetensors.safe_open(tensor_file, "numpy") as f:
        keys = f.keys()
        for tensor in utils.get_all_tensors(proto):
            name = tensor.name
            if name not in keys:
                continue
            place_holder = onnx.helper.make_tensor(
                name,
                tensor.data_type,
                tensor.dims,
                vals=f.get_tensor(name),
            )
            tensor.raw_data = place_holder.raw_data
            utils.set_external_data_flag(tensor, False)
            applied.add(name)
    return applied


def load(proto: ModelOrGraph, data: bytes) -> set[str]:
    """Load external data into ONNX model from safetensors bytes.

    Args:
        proto: ONNX model or graph to load external data into.
        data: safetensors bytes to load into ONNX model.

    Returns:
        Names of tensors that were applied.
    """
    tensor_dict = safetensors.numpy.load(data)
    return apply_tensors(proto, tensor_dict)


def save_file(
    proto: ModelOrGraph,
    tensor_file: str | os.PathLike,
    *,
    size_threshold: int = 0,
    convert_attributes: bool = False,
    strip_data: bool = False,
    matcher: Callable[[onnx.TensorProto], bool] | None = None,
) -> set[str]:
    """Save all tensors in an ONNX model to a safetensors file.

    Args:
        proto: ONNX model proto to save.
        tensor_file: Path to save the safetensors file.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        convert_attributes: If True, convert all tensors in attributes to safetensors.
            Otherwise, only convert initializer tensors.
        strip_data: If True, remove the tensor data from the ONNX model after saving.
            This will modify the ONNX model in place. Enable to preserve memory.
        matcher: A function that takes a TensorProto and returns True if the tensor
            should be saved. If None, all tensors are saved.

    Returns:
        A set of tensor names that were saved.
    """
    tensor_dict = _extract_tensors(
        proto,
        size_threshold=size_threshold,
        convert_attributes=convert_attributes,
        strip_data=strip_data,
        matcher=matcher,
    )

    safetensors.numpy.save_file(tensor_dict, tensor_file)
    return set(tensor_dict)


def save(
    proto: ModelOrGraph,
    *,
    size_threshold: int = 0,
    convert_attributes: bool = False,
    strip_data: bool = False,
    matcher: Callable[[onnx.TensorProto], bool] | None = None,
) -> tuple[bytes, set[str]]:
    """Save all tensors in an ONNX model to safetensors bytes.

    Args:
        proto: ONNX model proto to save.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        convert_attributes: If True, convert all tensors in attributes to safetensors.
            Otherwise, only convert initializer tensors.
        strip_data: If True, remove the tensor data from the ONNX model after saving.
            This will modify the ONNX model in place. Enable to preserve memory.
        matcher: A function that takes a TensorProto and returns True if the tensor
            should be saved. If None, all tensors are saved.

    Returns:
        A set of tensor names that were saved.
    """
    tensor_dict = _extract_tensors(
        proto,
        size_threshold=size_threshold,
        convert_attributes=convert_attributes,
        strip_data=strip_data,
        matcher=matcher,
    )

    return safetensors.numpy.save(tensor_dict), set(tensor_dict)
