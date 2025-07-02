"""Private module for loading and saving safetensors data to ONNX models."""

from __future__ import annotations

import io
import json
import math
import os
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

import onnx
import onnx.helper
import onnx_ir as ir
import safetensors

from onnx_safetensors import _tensors

if TYPE_CHECKING:
    pass


_HEADER_SIZE_NUMBER_SIZE = 8
# https://github.com/huggingface/safetensors/blob/543243c3017e413584f27ebd4b99c844f62deb34/safetensors/src/tensor.rs#L664
_SAFETENSORS_DTYPE_TO_IR_DTYPE = {
    "BOOL": ir.DataType.BOOL,
    "F8_E5M2": ir.DataType.FLOAT8E5M2,
    "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
    "BF16": ir.DataType.BFLOAT16,
    "F16": ir.DataType.FLOAT16,
    "F32": ir.DataType.FLOAT,
    "F64": ir.DataType.DOUBLE,
    "I8": ir.DataType.INT8,
    "I16": ir.DataType.INT16,
    "I32": ir.DataType.INT32,
    "I64": ir.DataType.INT64,
    "U8": ir.DataType.UINT8,
    "U16": ir.DataType.UINT16,
    "U32": ir.DataType.UINT32,
    "U64": ir.DataType.UINT64,
}
_IR_DTYPE_TO_SAFETENSORS_DTYPE = {
    ir.DataType.BOOL: "bool",
    ir.DataType.FLOAT4E2M1: "uint8",
    ir.DataType.FLOAT8E5M2: "float8_e5m2",
    ir.DataType.FLOAT8E4M3FN: "float8_e4m3fn",
    ir.DataType.FLOAT8E4M3FNUZ: "uint8",
    ir.DataType.FLOAT8E5M2FNUZ: "uint8",
    ir.DataType.BFLOAT16: "bfloat16",
    ir.DataType.FLOAT16: "float16",
    ir.DataType.FLOAT: "float32",
    ir.DataType.DOUBLE: "float64",
    ir.DataType.INT4: "uint8",
    ir.DataType.INT8: "int8",
    ir.DataType.INT16: "int16",
    ir.DataType.INT32: "int32",
    ir.DataType.INT64: "int64",
    ir.DataType.UINT4: "uint8",
    ir.DataType.UINT8: "uint8",
    ir.DataType.UINT16: "uint16",
    ir.DataType.UINT32: "uint32",
    ir.DataType.UINT64: "uint64",
}


TModel = TypeVar("TModel", onnx.ModelProto, ir.Model)


def _apply_tensors(
    model: ir.Model,
    tensors: Mapping[str, ir.TensorProtocol],
    apply_safetensors: bool = False,
):
    """Apply tensors to an ONNX model.

    Args:
        model: ONNX model to apply tensors to.
        tensors: Tensors to apply to the ONNX model.
        apply_safetensors: Whether it is applying safetensors to the ONNX model.
    """
    graph = model.graph
    for name, tensor in tensors.items():
        if name not in graph.initializers:
            continue
        model_tensor = graph.initializers[name].const_value
        if model_tensor is not None and apply_safetensors:
            assert isinstance(tensor, ir.ExternalTensor)
            _check_tensors_match(model_tensor, tensor)
            updated_tensor = _migrate_tensor_shape_dtype(model_tensor, tensor)
        else:
            updated_tensor = tensor
        graph.initializers[name].const_value = updated_tensor


def _is_4bit(dtype: ir.DataType) -> bool:
    return dtype in {
        ir.DataType.UINT4,
        ir.DataType.INT4,
        ir.DataType.FLOAT4E2M1,
    }


def _is_8bit_float(dtype: ir.DataType) -> bool:
    return dtype in {
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2FNUZ,
    }


def replace_tensors(
    model: ir.Model, /, location: str | os.PathLike, base_dir: str | os.PathLike
) -> None:
    """Replace all tensors in an ONNX model with external data from a safetensors file.

    Args:
        model: ONNX model to replace tensors in.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.

    .. versionadded:: 1.0
        Added the function.
    """
    tensors = _read_safetensors(location, base_dir=base_dir)
    _apply_tensors(model, tensors, apply_safetensors=True)


def load_file(model: TModel, /, tensor_file: str | os.PathLike) -> TModel:
    """Load external data into ONNX model from a safetensors file.

    Args:
        model: ONNX model.
        tensor_file: safetensors file to load into ONNX model.

    .. versionchanged:: 1.0
        The return value is now the updated ONNX model instead of a set of loaded tensor names.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    replace_tensors(model_ir, tensor_file, "")
    model_ir = ir.external_data.load_to_model(model_ir)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def load(model: TModel, /, data: bytes) -> TModel:
    """Load external data into ONNX model from safetensors bytes.

    Args:
        model: ONNX model.
        data: safetensors bytes to load into ONNX model.

    .. versionchanged:: 1.0
        The return value is now the updated ONNX model instead of a set of loaded tensor names.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    # TODO: Handle more dtypes
    tensors = safetensors.deserialize(data)
    tensors_dict = {
        name: _tensors.ByteArrayTensor(
            data=metadata["data"],
            dtype=_SAFETENSORS_DTYPE_TO_IR_DTYPE[metadata["dtype"]],
            shape=ir.Shape(metadata["shape"]),
            name=name,
        )
        for (name, metadata) in tensors
    }
    _apply_tensors(model_ir, tensors_dict)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def load_file_as_external_data(
    model: TModel, /, location: str | os.PathLike, base_dir: str | os.PathLike = ""
) -> TModel:
    """Load weights from safetensors file and use them as external data for the ONNX model.

    Args:
        model: ONNX model or graph to load external data into.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.

    Returns:
        The ONNX model with the external data.

    .. versionadded:: 1.0
        Added the function.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    replace_tensors(model_ir, location, base_dir)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def _get_tensor_storage_shape(tensor: ir.TensorProtocol) -> list[int]:
    if _is_4bit(tensor.dtype):
        return [math.ceil(math.prod(tensor.shape.numpy()) / 2)]
    return tensor.shape.numpy()


def save(model: TModel, /, *, size_threshold: int = 0) -> bytes:
    """Save all tensors in an ONNX model to a safetensors object serialized as bytes.

    Args:
        model: ONNX model to save.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all initializers.

    Returns:
        The safetensors object serialized as bytes.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    tensor_dict: dict[str, dict[str, Any]] = {}
    for name, initializer in model_ir.graph.initializers.items():
        if initializer.const_value is None:
            continue
        if initializer.const_value.size < size_threshold:
            continue
        tensor = initializer.const_value
        tensor_dict[name] = {
            "dtype": _IR_DTYPE_TO_SAFETENSORS_DTYPE[tensor.dtype],
            "shape": _get_tensor_storage_shape(tensor),
            # TODO: Return a memoryview when safetensors supports it.
            "data": tensor.tobytes(),
        }
    return safetensors.serialize(tensor_dict)


def save_file(
    model: TModel,
    /,
    location: str | os.PathLike,
    base_dir: str | os.PathLike = "",
    *,
    size_threshold: int = 0,
    replace_data: bool = True,
) -> TModel:
    """Save all tensors in an ONNX model to a safetensors file.

    Args:
        model: ONNX model proto to save.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        replace_data: Whether to replace the data in the ONNX model with
            the external data. Default is True.

    Returns:
        The ONNX model with the external data.

    .. versionadded:: 1.0.1
        The *base_dir* parameter was added so the external data can be referenced
        relative to the ONNX model file correctly.
    .. versionadded:: 1.0
        The *replace_data* parameter was added to allow the user to choose
        whether to replace the data in the ONNX model with the external data.
    .. versionremoved:: 1.0
        The *convert_attributes* and *strip_data* parameters were removed. Set
        *replace_data* to achieve similar effect as *strip_data*.
    .. versionchanged:: 1.0
        The return value is now the updated ONNX model instead of a set of saved tensor names.
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
        tensor = initializer.const_value
        tensor_dict[name] = {
            "dtype": _IR_DTYPE_TO_SAFETENSORS_DTYPE[tensor.dtype],
            "shape": _get_tensor_storage_shape(tensor),
            # TODO: Return a memoryview when safetensors supports it.
            "data": tensor.tobytes(),
        }
    tensor_file = os.path.join(base_dir, location)
    safetensors.serialize_file(tensor_dict, tensor_file)
    if replace_data:
        replace_tensors(model_ir, location, base_dir)

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
    location: str | os.PathLike, base_dir: str | os.PathLike
) -> dict[str, ir.ExternalTensor]:
    """Read a safetensors file.

    Args:
        location: The safetensors file to read.
        base_dir: Directory where the ONNX model file is stored.

    Returns:
        The contents of the safetensors file.
    """
    path = os.path.join(base_dir, location)
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
            dtype=_SAFETENSORS_DTYPE_TO_IR_DTYPE[metadata["dtype"]],
            shape=ir.Shape(metadata["shape"]),
            name=name,
            base_dir=base_dir,
        )
    return tensors


def _check_tensors_match(
    model_tensor: ir.TensorProtocol, safe_tensor: ir.ExternalTensor
):
    """Check if two tensors match.

    Args:
        model_tensor: Tensor from the model.
        safe_tensor: Tensor from the safetensors file.

    Raises:
        ValueError: If the tensors do not match.
    """
    if _is_4bit(model_tensor.dtype):
        if safe_tensor.dtype != ir.DataType.UINT8:
            raise ValueError(
                f"The tensor from safetensors has dtype: {safe_tensor.dtype}, but it must be UINT8 to "
                f"represent the dtype of the tensor in the model: {model_tensor.dtype}."
            )
        if model_tensor.nbytes != safe_tensor.nbytes:
            raise ValueError(
                f"The tensor from safetensors has size: {safe_tensor.nbytes} bytes, "
                f"which does not match the size of the tensor in the model: {model_tensor.nbytes} bytes."
            )
        return

    if _is_8bit_float(model_tensor.dtype):
        if (
            not _is_8bit_float(safe_tensor.dtype)
            and safe_tensor.dtype != ir.DataType.UINT8
        ):
            raise ValueError(
                f"The tensor from safetensors has dtype: {safe_tensor.dtype}, but it must be UINT8 to "
                f"represent the dtype of the tensor in the model: {model_tensor.dtype}."
            )
    elif model_tensor.dtype != safe_tensor.dtype:
        raise ValueError(
            f"The tensor from safetensors has dtype: {safe_tensor.dtype}, "
            f"which does not match the dtype of the tensor in the model: {model_tensor.dtype}."
        )

    if model_tensor.shape != safe_tensor.shape:
        raise ValueError(
            f"The tensor from safetensors has shape: {safe_tensor.shape}, "
            f"which does not match the shape of the tensor in the model: {model_tensor.shape}."
        )


def _migrate_tensor_shape_dtype(
    model_tensor: ir.TensorProtocol, safe_tensor: ir.ExternalTensor
) -> ir.ExternalTensor:
    """Migrate the shape and dtype of a tensor.

    Args:
        model_tensor: The tensor to migrate.
        safe_tensor: The tensor to migrate to.

    Returns:
        The migrated tensor.
    """
    if model_tensor.dtype in {
        # ir.DataType.FLOAT8E4M3FN,
        # ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2FNUZ,
    } or _is_4bit(model_tensor.dtype):
        return ir.ExternalTensor(
            location=safe_tensor.location,
            offset=safe_tensor.offset,
            length=safe_tensor.length,
            dtype=model_tensor.dtype,
            shape=model_tensor.shape,
            name=safe_tensor.name,
            base_dir=safe_tensor.base_dir,
        )
    return safe_tensor
