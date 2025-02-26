"""Private module for loading and saving safetensors data to ONNX models."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Union, TypeVar

import onnx
import onnx.helper
import safetensors
import safetensors.numpy

from onnxscript import ir

if TYPE_CHECKING:
    import numpy as np


_SAFETENSORS_TYPE_TO_IR_TYPE = {
    "BOOL": ir.DataType.BOOL,
    "F8_E5M2": ir.DataType.FLOAT8E5M2,
    "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
    "BF16": ir.DataType.BFLOAT16,
    "F16": ir.DataType.FLOAT16,
    "F32": ir.DataType.FLOAT,
    "F64": ir.DataType.DOUBLE,
    "I4": ir.DataType.INT4,
    "I8": ir.DataType.INT8,
    "I16": ir.DataType.INT16,
    "I32": ir.DataType.INT32,
    "I64": ir.DataType.INT64,
    "U4": ir.DataType.UINT4,
    "U8": ir.DataType.UINT8,
    "U16": ir.DataType.UINT16,
    "U32": ir.DataType.UINT32,
    "U64": ir.DataType.UINT64,
}
_HEADER_SIZE_NUMBER_SIZE = 8


TModel = TypeVar("TModel", onnx.ModelProto, ir.Model)


def _apply_tensors(model: ir.Model, tensors: Mapping[str, ir.TensorProtocol]):
    """Apply tensors to an ONNX model.

    Args:
        model: ONNX model to apply tensors to.
        tensors: Tensors to apply to the ONNX model.
    """
    graph = model.graph
    for name, tensor in tensors.items():
        if name not in graph.initializers:
            continue
        graph.initializers[name].const_value = tensor


def replace_tensors(
    model: ir.Model, location: str | os.PathLike, base_path: str | os.PathLike
) -> None:
    """Replace all tensors in an ONNX model with external data from a safetensors file.

    Args:
        proto: ONNX model or graph to replace tensors in.
        location: Path to the safetensors file.
        base_path: Base path for the safetensors file.
    """
    tensors = _read_safetensors(location, base_path)
    _apply_tensors(model, tensors)


def load_file(model: TModel, tensor_file: str | os.PathLike) -> TModel:
    """Load external data into ONNX model from a safetensors file.

    Args:
        model: ONNX model.
        tensor_file: safetensors file to load into ONNX model.
    """
    # TODO(justinchuby): Handle safetensors unsupported dtypes
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    replace_tensors(model_ir, tensor_file, "")
    model_ir = ir.external_data.load_to_model(model_ir)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def load(model: TModel, data: bytes) -> TModel:
    """Load external data into ONNX model from safetensors bytes.

    Args:
        model: ONNX model.
        data: safetensors bytes to load into ONNX model.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    # TODO: Handle more dtypes
    tensors = safetensors.numpy.load(data)
    tensors_dict = {
        name: ir.Tensor(value=value, name=name) for name, value in tensors.items()
    }
    _apply_tensors(model_ir, tensors_dict)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def load_file_as_external_data(
    model: TModel, location: str | os.PathLike, base_path: str | os.PathLike = ""
) -> TModel:
    """Load weights from safetensors file and use them as external data for the ONNX model.

    Args:
        model: ONNX model or graph to load external data into.
        location: Path to the safetensors file.
        base_path: Base path for the safetensors file.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    replace_tensors(model_ir, location, base_path)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def save_file(
    model: TModel,
    location: str | os.PathLike,
    base_path: str | os.PathLike = "",
    *,
    size_threshold: int = 0,
    replace_data: bool = True,
) -> TModel:
    """Save all tensors in an ONNX model to a safetensors file.

    Args:
        proto: ONNX model proto to save.
        location: Relative path to the safetensors file.
        base_path: Base path for the safetensors file.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        replace_data: Whether to replace the data in the ONNX model with
            the external data. Default is True.

    Returns:
        The ONNX model with the external data.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    tensor_dict = {}
    for name, initializer in model_ir.graph.initializers.items():
        if initializer.const_value is None:
            continue
        if initializer.const_value.size < size_threshold:
            continue
        tensor_dict[name] = initializer.const_value.numpy()

    tensor_file = os.path.join(base_path, location)
    safetensors.numpy.save_file(tensor_dict, tensor_file)
    if replace_data:
        replace_tensors(model_ir, location, base_path)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def _read_safetensors_header(file: io.IOBase) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    file.seek(0)
    header_size = struct.unpack_from("i", file.read(_HEADER_SIZE_NUMBER_SIZE))[0]
    header = file.read(header_size)
    return json.loads(header.decode("utf-8")), header_size


def _read_safetensors(
    location: str | os.PathLike, base_path: str | os.PathLike
) -> dict[str, ir.ExternalTensor]:
    """Read a safetensors file.

    Args:
        location: The safetensors file to read.
        base_path: Base path for the safetensors file.

    Returns:
        The contents of the safetensors file.
    """
    path = os.path.join(base_path, location)
    with open(path, "rb") as file:
        header, header_size = _read_safetensors_header(file)
    tensors = {}
    for name, metadata in header.items():
        offset = metadata["data_offsets"][0] + header_size + _HEADER_SIZE_NUMBER_SIZE
        length = metadata["data_offsets"][1] - metadata["data_offsets"][0]
        tensors[name] = ir.ExternalTensor(
            location=location,
            offset=offset,
            length=length,
            dtype=_SAFETENSORS_TYPE_TO_IR_TYPE[metadata["dtype"]],
            shape=ir.Shape(metadata["shape"]),
            name=name,
            base_dir=base_path,
        )
    return tensors
