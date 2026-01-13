"""Private module for loading and saving safetensors data to ONNX models."""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import onnx
import onnx_ir as ir
import safetensors
from tqdm.auto import tqdm

from onnx_safetensors import _tensors

if TYPE_CHECKING:
    pass


_HEADER_SIZE_NUMBER_SIZE = 8
# https://github.com/huggingface/safetensors/blob/543243c3017e413584f27ebd4b99c844f62deb34/safetensors/src/tensor.rs#L664
_SAFETENSORS_DTYPE_TO_IR_DTYPE = {
    "BOOL": ir.DataType.BOOL,
    "F4": ir.DataType.FLOAT4E2M1,
    "F8_E5M2": ir.DataType.FLOAT8E5M2,
    "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
    "F8_E8M0": ir.DataType.FLOAT8E8M0,
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
    "C64": ir.DataType.COMPLEX64,
}
_IR_DTYPE_TO_SAFETENSORS_DTYPE = {
    ir.DataType.BOOL: "bool",
    ir.DataType.FLOAT4E2M1: "float4_e2m1fn_x2",
    ir.DataType.FLOAT8E5M2: "float8_e5m2",
    ir.DataType.FLOAT8E4M3FN: "float8_e4m3fn",
    ir.DataType.FLOAT8E8M0: "float8_e8m0",
    ir.DataType.FLOAT8E4M3FNUZ: "uint8",
    ir.DataType.FLOAT8E5M2FNUZ: "uint8",
    ir.DataType.BFLOAT16: "bfloat16",
    ir.DataType.FLOAT16: "float16",
    ir.DataType.FLOAT: "float32",
    ir.DataType.DOUBLE: "float64",
    ir.DataType.INT2: "uint8",
    ir.DataType.INT4: "uint8",
    ir.DataType.INT8: "int8",
    ir.DataType.INT16: "int16",
    ir.DataType.INT32: "int32",
    ir.DataType.INT64: "int64",
    ir.DataType.UINT2: "uint8",
    ir.DataType.UINT4: "uint8",
    ir.DataType.UINT8: "uint8",
    ir.DataType.UINT16: "uint16",
    ir.DataType.UINT32: "uint32",
    ir.DataType.UINT64: "uint64",
    ir.DataType.COMPLEX64: "complex64",
}


TModel = TypeVar("TModel", onnx.ModelProto, ir.Model)


def _parse_size_string(size: int | str) -> int:
    """Parse a size string like '5GB' or '100MB' into bytes.

    Args:
        size: Either an integer representing bytes, or a string like '5GB', '100MB', etc.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the size string format is invalid.
    """
    if isinstance(size, int):
        return size

    size = size.strip()
    match = re.match(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", size)
    if not match:
        raise ValueError(
            f"Invalid size format: {size}. Expected format like '5GB' or '100MB'."
        )

    num_str, unit = match.groups()
    num = float(num_str)

    # Convert to bytes
    unit = unit.upper()
    multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
    }

    if unit not in multipliers:
        raise ValueError(
            f"Unknown size unit: {unit}. Valid units are: {', '.join(multipliers.keys())}"
        )

    return int(num * multipliers[unit])


def _get_shard_filename(base_name: str, shard_idx: int, total_shards: int) -> str:
    """Generate a filename for a shard.

    Args:
        base_name: The base filename (e.g., 'model.safetensors').
        shard_idx: The index of this shard (1-indexed).
        total_shards: The total number of shards.

    Returns:
        The shard filename (e.g., 'model-00001-of-00003.safetensors').
    """
    if total_shards == 1:
        return base_name

    # Extract extension
    if "." in base_name:
        name, ext = base_name.rsplit(".", 1)
        ext = f".{ext}"
    else:
        name = base_name
        ext = ""

    # Always use 5 digits to follow transformers convention
    return f"{name}-{shard_idx:05d}-of-{total_shards:05d}{ext}"


def _shard_tensors(
    tensors: Sequence[ir.TensorProtocol], max_shard_size_bytes: int | None
) -> list[list[ir.TensorProtocol]]:
    """Shard tensors into multiple files based on max_shard_size_bytes.

    Args:
        tensors: The tensors to shard.
        max_shard_size_bytes: Maximum size for each shard in bytes. When None,
            no sharding is performed.

    Returns:
        A list of tensor lists for each shard.
    """
    if max_shard_size_bytes is None:
        # No sharding
        return [list(tensors)]

    # Shard the tensors by current order
    shards: list[list[ir.TensorProtocol]] = [[]]
    current_shard_size = 0

    for tensor in tensors:
        tensor_size = tensor.nbytes
        # Check if adding this tensor would exceed max_shard_size_bytes
        if (
            current_shard_size + tensor_size > max_shard_size_bytes
            and current_shard_size > 0
        ):
            # Start a new shard
            shards.append([])
            current_shard_size = 0

        shards[-1].append(tensor)
        current_shard_size += tensor_size

    return shards


def _apply_tensors(
    values: Sequence[ir.Value],
    tensors: Mapping[str, ir.TensorProtocol],
    apply_safetensors: bool = False,
):
    """Apply tensors to an ONNX model.

    Args:
        values: Values in the ONNX model.
        tensors: Tensors to apply to the ONNX model.
        apply_safetensors: Whether it is applying safetensors to the ONNX model.
    """
    value_map: dict[str, ir.Value] = {value.name: value for value in values}  # type: ignore[misc]
    for name, tensor in tensors.items():
        if name not in value_map:
            continue
        value = value_map[name]
        model_tensor = value_map[name].const_value
        if model_tensor is not None and apply_safetensors:
            assert isinstance(tensor, ir.ExternalTensor)
            _check_tensors_match(model_tensor, tensor)
            updated_tensor = _migrate_tensor_shape_dtype(model_tensor, tensor)
        else:
            updated_tensor = tensor
        value.const_value = updated_tensor


def replace_tensors(
    model: ir.Model, /, location: str | os.PathLike, base_dir: str | os.PathLike
) -> None:
    """Replace all tensors in an ONNX model with external data from a safetensors file.

    .. versionadded:: 1.0
        Added the function.

    Args:
        model: ONNX model to replace tensors in.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
    """
    tensors = _read_safetensors(location, base_dir=base_dir)
    values = [value for value, _ in _get_value_tensor_pairs(model)]
    _apply_tensors(values, tensors, apply_safetensors=True)


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
    values = [value for value, _ in _get_value_tensor_pairs(model_ir)]
    _apply_tensors(values, tensors_dict)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def load_file_as_external_data(
    model: TModel, /, location: str | os.PathLike, base_dir: str | os.PathLike = ""
) -> TModel:
    """Load weights from safetensors file and use them as external data for the ONNX model.

    .. versionadded:: 1.0
        Added the function.

    Args:
        model: ONNX model or graph to load external data into.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.

    Returns:
        The ONNX model with the external data.
    """
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    replace_tensors(model_ir, location, base_dir)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def _get_tensor_storage_shape(tensor: ir.TensorProtocol) -> Sequence[int]:
    """Get the storage shape of a tensor for safetensors."""
    # Handle sub-byte dtypes
    if tensor.dtype.bitwidth < 8:
        return [tensor.nbytes]
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


def _get_value_tensor_pairs(
    model: ir.Model,
) -> list[tuple[ir.Value, ir.TensorProtocol]]:
    # Store the original initializer values so they can be restored if modify_model=False
    value_tensor_pairs: list[tuple[ir.Value, ir.TensorProtocol]] = []
    initializer_names: set[str] = set()
    for graph in model.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            # The value.name should be the same as tensor.name. However,
            # in case there is a conflict, we do not care and will prefer value.name.
            name = value.name
            if name is None:
                raise ValueError(
                    f"Initializer value '{value!r}' has no name (in graph {graph.name!r}). "
                    "All initializers must have names."
                )
            if tensor is None:
                continue
            if name in initializer_names:
                raise ValueError(
                    f"Duplicate initializer name found: {name} (in graph {graph.name!r})."
                    " Rename the initializers to have unique names before saving to safetensors."
                )
            initializer_names.add(name)
            value_tensor_pairs.append((value, tensor))
    return value_tensor_pairs


def save_file(  # noqa: PLR0912, PLR0915
    model: TModel,
    /,
    location: str | os.PathLike,
    base_dir: str | os.PathLike = "",
    *,
    size_threshold: int = 0,
    replace_data: bool = True,
    max_shard_size: int | str | None = None,
) -> TModel:
    """Save all tensors in an ONNX model to a safetensors file.

    .. versionadded:: 1.0.0
        The *replace_data* parameter was added to allow the user to choose
        whether to replace the data in the ONNX model with the external data.
    .. versionremoved:: 1.0.0
        The *convert_attributes* and *strip_data* parameters were removed. Set
        *replace_data* to achieve similar effect as *strip_data*.
    .. versionchanged:: 1.0.0
        The return value is now the updated ONNX model instead of a set of saved tensor names.
    .. versionadded:: 1.0.1
        The *base_dir* parameter was added so the external data can be referenced
        relative to the ONNX model file correctly.
    .. versionadded:: 1.3.0
        The *max_shard_size* parameter was added to support sharding large models.

    Args:
        model: ONNX model proto to save.
        location: Path to the safetensors file relative to the ONNX model file.
        base_dir: Directory where the ONNX model file is stored.
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        replace_data: Whether to replace the data in the ONNX model with
            the external data. Default is True.
        max_shard_size: Maximum size in bytes (as int) or as a string with unit
            (like "5GB" or "100MB") for a checkpoint before being sharded.
            If None, no sharding is performed.

    Returns:
        The ONNX model with the external data.
    """
    # Ensure that external_data ends with .safetensors
    if not str(location).endswith(".safetensors"):
        raise ValueError(
            f'The path to safetensors file must have a .safetensors extension, got: "{location}"'
        )

    max_shard_size_bytes = (
        _parse_size_string(max_shard_size) if max_shard_size is not None else None
    )
    size_threshold_bytes = size_threshold

    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model
    initialized_values = [value for value, _ in _get_value_tensor_pairs(model_ir)]
    # First, collect metadata without loading tensor data
    tensors_to_save: list[ir.TensorProtocol] = []
    values_to_save: list[ir.Value] = []
    for value in initialized_values:
        tensor = value.const_value
        assert tensor is not None
        if tensor.nbytes < size_threshold_bytes:
            continue
        tensors_to_save.append(tensor)
        values_to_save.append(value)

    total_size = sum(tensor.nbytes for tensor in tensors_to_save)

    if tensors_to_save:
        # Determine sharding based on max_shard_size_bytes. When max_shard_size_bytes is None,
        # It is the same as one shard (which is the same as no sharding).
        tensor_shards = _shard_tensors(tensors_to_save, max_shard_size_bytes)
        total_shards = len(tensor_shards)

        # Save each shard, loading only necessary tensor data
        all_filenames = []
        weight_map: dict[str, str] = {}  # Maps tensor name to shard filename
        current_offset = 0
        current_index = 0
        for shard_idx, tensor_shard in enumerate(tensor_shards, start=1):
            shard_filename = _get_shard_filename(str(location), shard_idx, total_shards)

            shard_path = os.path.join(base_dir, shard_filename)
            all_filenames.append(shard_filename)

            # Build tensor_dict for this shard only
            shard_dict: dict[str, Any] = {}
            for tensor in tensor_shard:
                assert tensor.name is not None
                shard_dict[tensor.name] = {
                    "dtype": _IR_DTYPE_TO_SAFETENSORS_DTYPE[tensor.dtype],
                    "shape": _get_tensor_storage_shape(tensor),
                    "data": tensor.tobytes(),
                }
                # Update weight_map with shard filename
                weight_map[tensor.name] = shard_filename
                current_offset += tensor.nbytes
                current_index += 1

            safetensors.serialize_file(shard_dict, shard_path)

        # Save index file if sharding occurred
        if total_shards > 1:
            location_str = str(location)
            if location_str.endswith(".safetensors"):
                index_filename = (
                    location_str.rsplit(".safetensors", 1)[0]
                    + ".safetensors.index.json"
                )
            else:
                index_filename = location_str + ".index.json"
            index_path = os.path.join(base_dir, index_filename)
            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        # Replace tensors from each shard file
        if replace_data:
            for filename in all_filenames:
                replace_tensors(model_ir, filename, base_dir)

    if isinstance(model, onnx.ModelProto):
        return ir.serde.serialize_model(model_ir)
    return model_ir


def save_model(
    model: TModel,
    model_path: str | os.PathLike,
    /,
    *,
    external_data: str | os.PathLike | None = None,
    size_threshold: int = 0,
    max_shard_size: int | str | None = None,
) -> None:
    """Save an ONNX model to a file with external data in a safetensors file.

    .. versionadded:: 1.3.0
        Added the function.

    Args:
        model: ONNX model to save.
        model_path: Path to the ONNX model file. E.g. "model.onnx".
        external_data: Path to the safetensors file relative to the ONNX model file.
            E.g. "model.safetensors". If not provided, it will be derived from
            the model_path by replacing the extension with ".safetensors".
        size_threshold: Minimum size in bytes for a tensor to be saved.
            Default is 0, which saves all tensors.
        max_shard_size: Maximum size in bytes for a checkpoint before being sharded.
            If expressed as a string, needs to be digits followed by a unit
            (like "5GB" or "100MB"). If None, no sharding is performed.
            When sharding is enabled, multiple safetensors files will be created
            with names like "model-00001-of-00003.safetensors", and an index
            file "model.safetensors.index.json" will be created to map tensors
            to their respective shard files.

    Raises:
        ValueError: If external_data does not end with ".safetensors".
    """
    # Derive external_data from model_path if not provided
    if external_data is None:
        model_path_str = str(model_path)
        # Get the base name without extension
        if "." in os.path.basename(model_path_str):
            base_name = os.path.splitext(os.path.basename(model_path_str))[0]
        else:
            base_name = os.path.basename(model_path_str)
        external_data = f"{base_name}.safetensors"

    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
    else:
        model_ir = model

    # Store the original initializer values so they can be restored if modify_model=False
    value_tensor_pairs = _get_value_tensor_pairs(model_ir)

    try:
        save_file(
            model_ir,
            external_data,
            os.path.dirname(model_path),
            size_threshold=size_threshold,
            max_shard_size=max_shard_size,
        )
        ir.save(model_ir, model_path)
    finally:
        # Restore original initializers to avoid side effects
        for value, tensor in value_tensor_pairs:
            value.const_value = tensor


def _read_safetensors_header(file: io.IOBase) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    file.seek(0)
    header_size = struct.unpack_from("<Q", file.read(_HEADER_SIZE_NUMBER_SIZE))[0]
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
        if name == "__metadata__":
            continue
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
    if model_tensor.nbytes != safe_tensor.nbytes:
        raise ValueError(
            f"Tensor size mismatch for tensor '{model_tensor.name}': "
            f"model tensor size {model_tensor.nbytes} bytes, "
            f"safetensors tensor size {safe_tensor.nbytes} bytes. "
            f"Model tensor: {model_tensor}, Safetensors tensor: {safe_tensor}"
        )


def _migrate_tensor_shape_dtype(
    model_tensor: ir.TensorProtocol, safe_tensor: ir.ExternalTensor
) -> ir.ExternalTensor:
    """Migrate the shape and dtype of a tensor.

    This is needed because we store 4bit and 2bit tensors as UINT8 in safetensors.

    Args:
        model_tensor: The tensor to migrate.
        safe_tensor: The tensor to migrate to.

    Returns:
        The migrated tensor.
    """
    if model_tensor.dtype in {
        # Types that safetensors does not support directly
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2FNUZ,
        ir.DataType.FLOAT4E2M1,  # Still need to migrate shape
        ir.DataType.INT4,
        ir.DataType.INT2,
        ir.DataType.UINT4,
        ir.DataType.UINT2,
    }:
        return ir.ExternalTensor(
            location=safe_tensor.location,
            offset=safe_tensor.offset,
            length=safe_tensor.length,
            dtype=model_tensor.dtype,
            shape=model_tensor.shape,  # type: ignore[arg-type]
            name=safe_tensor.name,
            base_dir=safe_tensor.base_dir,
        )
    return safe_tensor
