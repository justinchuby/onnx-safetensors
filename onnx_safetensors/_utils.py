# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Utilities for iterating on ONNX models."""

from __future__ import annotations
from io import BytesIO

__all__ = []

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


def read_safetensors_header(file: io.IOBase) -> dict[str, dict[str, Any]]:
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    file.seek(0)
    header_size = struct.unpack_from("i", file.read(8))[0]
    header = file.read(header_size)
    return json.loads(header.decode("utf-8"))


f = open("dd.safetensors", "rb")
print(read_safetensors_header(f))