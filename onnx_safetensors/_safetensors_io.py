from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import onnx
import safetensors
import safetensors.numpy

from onnx_safetensors import _external_data_helper


def load_file(model_proto: onnx.ModelProto, tensor_file: str | os.PathLike) -> None:
    """Load external data into ONNX model from a safetensors file.

    Args:
        model_proto: ONNX model to load external data into.
        tensor_file: safetensors file to load into ONNX model.
    """
    with safetensors.safe_open(tensor_file, "numpy") as f:
        keys = f.keys()
        for tensor in _external_data_helper.get_all_tensors(model_proto):
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
            _external_data_helper.set_external_data_flag(tensor, False)


def load(model_proto: onnx.ModelProto, data: bytes) -> None:
    """Load external data into ONNX model from safetensors bytes.

    Args:
        model_proto: ONNX model to load external data into.
        data: safetensors bytes to load into ONNX model.
    """
    tensor_dict = safetensors.numpy.load(data)

    for tensor in _external_data_helper.get_all_tensors(model_proto):
        name = tensor.name
        if (external_tensor := tensor_dict.get(name)) is None:
            continue
        place_holder = onnx.helper.make_tensor(
            name,
            tensor.data_type,
            tensor.dims,
            vals=external_tensor,
        )
        tensor.raw_data = place_holder.raw_data
        _external_data_helper.set_external_data_flag(tensor, False)


def _extract_tensors(
    model_proto: onnx.ModelProto,
    size_threshold: int = 0,
    convert_attributes: bool = False,
    strip_data: bool = False,
    matcher: Callable[[onnx.TensorProto], bool] | None = None,
) -> dict[str, np.ndarray]:
    if convert_attributes:
        tensors = _external_data_helper.get_all_tensors(model_proto)
    else:
        tensors = _external_data_helper.get_initializer_tensors(model_proto)

    tensor_dict = {}

    for tensor in tensors:
        name = tensor.name
        if not (
            tensor.HasField("raw_data")
            and sys.getsizeof(tensor.raw_data) >= size_threshold
        ):
            continue
        if matcher is not None and not matcher(tensor):
            continue
        try:
            tensor_dict[name] = onnx.numpy_helper.to_array(tensor)
        except Exception as e:  # noqa: PERF203
            raise RuntimeError(
                f"Failed to convert tensor '{name}' to numpy array."
            ) from e
        if strip_data:
            _external_data_helper.set_external_data_flag(tensor, True)

    return tensor_dict


def save_file(
    model_proto: onnx.ModelProto,
    tensor_file: str | os.PathLike,
    size_threshold: int = 0,
    convert_attributes: bool = False,
    strip_data: bool = False,
    matcher: Callable[[onnx.TensorProto], bool] | None = None,
) -> set[str]:
    """Save all tensors in an ONNX model to a safetensors file.

    Args:
        model_proto: ONNX model proto to save.
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
        model_proto,
        size_threshold=size_threshold,
        convert_attributes=convert_attributes,
        strip_data=strip_data,
        matcher=matcher,
    )

    safetensors.numpy.save_file(tensor_dict, tensor_file)
    return set(tensor_dict)


def save(
    model_proto: onnx.ModelProto,
    size_threshold: int = 0,
    convert_attributes: bool = False,
    strip_data: bool = False,
    matcher: Callable[[onnx.TensorProto], bool] | None = None,
) -> tuple[bytes, set[str]]:
    """Save all tensors in an ONNX model to safetensors bytes.

    Args:
        model_proto: ONNX model proto to save.
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
        model_proto,
        size_threshold=size_threshold,
        convert_attributes=convert_attributes,
        strip_data=strip_data,
        matcher=matcher,
    )

    return safetensors.numpy.save(tensor_dict), set(tensor_dict)


def strip_raw_data(model_proto: onnx.ModelProto, names: set[str]):
    """Remove raw tensor data from the ONNX model."""
    for tensor in _external_data_helper.get_all_tensors(model_proto):
        if tensor.name in names:
            _external_data_helper.set_external_data_flag(tensor, True)
