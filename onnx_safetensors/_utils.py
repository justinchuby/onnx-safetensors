# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Utilities for iterating on ONNX models."""

from __future__ import annotations

__all__ = [
    "get_attributes",
    "get_initializers",
]

from collections.abc import Iterable
from typing import Callable

from onnxscript import ir
import safetensors


# Loading safetensors
# 1. Load the offset for each tensor, and create an External Tensor for it
# to replace the initializers
# Saving safetensors
# 1. Load the initializer if any one of them are external.
# 2. For each initializer, save it into a safetensor file, then replace the initializer
