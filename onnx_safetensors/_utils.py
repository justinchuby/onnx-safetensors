# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Utilities for iterating on ONNX models."""

from __future__ import annotations
from io import BytesIO

__all__ = []

import os
import struct

from collections.abc import Iterable
from typing import Any, Callable

from onnxscript import ir
import safetensors
import json
import io

# Loading safetensors
# 1. Load the offset for each tensor, and create an External Tensor for it
# to replace the initializers
# Saving safetensors
# 1. Load the initializer if any one of them are external.
# 2. For each initializer, save it into a safetensor file, then replace the initializer

_SAFETENSORS_TYPE_TO_IR_TYPE = {
    "B": ir.DataType.BOOL,
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


def read_safetensors_header(file: io.IOBase) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    file.seek(0)
    header_size = struct.unpack_from("i", file.read(8))[0]
    header = file.read(header_size)
    return json.loads(header.decode("utf-8")), header_size


def read_safetensors(
    location: str | os.PathLike, base_path: str | os.PathLike
) -> dict[str, ir.ExternalTensor]:
    """Read a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The contents of the safetensors file.
    """
    path = os.path.join(base_path, location)
    with open(path, "rb") as file:
        header, header_size = read_safetensors_header(file)
    tensors = {}
    for name, metadata in header.items():
        tensors[name] = ir.ExternalTensor(
            location=location,
            offset=metadata["data_offsets"][0] + header_size + 8,
            length=metadata["data_offsets"][1]
            - metadata["data_offsets"][0]
            + header_size
            + 8,
            dtype=_SAFETENSORS_TYPE_TO_IR_TYPE[metadata["dtype"]],
            shape=ir.Shape(metadata["shape"]),
            name=name,
            base_dir=base_path,
        )
    return tensors


f = open("dd.safetensors", "rb")
print(read_safetensors_header(f))
