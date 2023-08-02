# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Utilities for iterating on ONNX models."""

from __future__ import annotations

__all__ = [
    "get_all_tensors",
    "get_attribute_tensors",
    "get_initializer_tensors",
    "set_external_data_flag",
    "apply_tensor_dict",
]

import itertools
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import onnx

if TYPE_CHECKING:
    import numpy as np


def _recursive_attribute_processor(
    attribute: onnx.AttributeProto,
    func: Callable[[onnx.GraphProto], Iterable[onnx.TensorProto]],
) -> Iterable[onnx.TensorProto]:
    """Create an iterator through processing ONNX model attributes with functor."""
    if attribute.type == onnx.AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == onnx.AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)


def get_initializer_tensors(
    proto: onnx.ModelProto | onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of initializer tensors from ONNX ModelProto or GraphProto."""
    if isinstance(proto, onnx.ModelProto):
        graph = proto.graph
    else:
        graph = proto
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(
                attribute, get_initializer_tensors
            )


def get_attribute_tensors(
    proto: onnx.ModelProto | onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX ModelProto or GraphProto."""
    if isinstance(proto, onnx.ModelProto):
        graph = proto.graph
    else:
        graph = proto
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(attribute, get_attribute_tensors)


def get_all_tensors(
    proto: onnx.ModelProto | onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return itertools.chain(
        get_initializer_tensors(proto),
        get_attribute_tensors(proto),
    )


def set_external_data_flag(tensor: onnx.TensorProto, flag: bool) -> None:
    """Set or unset the external data flag of a tensor."""
    # We do not need the metadata about external data
    del tensor.external_data[:]
    if flag:
        # After loading raw_data from external_data, change the state of tensors
        tensor.data_location = onnx.TensorProto.EXTERNAL
    else:
        tensor.data_location = onnx.TensorProto.DEFAULT


def clear_raw_data(tensor: onnx.TensorProto):
    """Clear raw_data of a tensor."""
    if tensor.HasField("raw_data"):
        tensor.ClearField("raw_data")


def apply_tensor_dict(
    tensor_protos: Iterable[onnx.TensorProto], tensor_dict: Mapping[str, np.ndarray]
) -> set[str]:
    """Apply a dictionary of external data to a list of `TensorProto`s.

    Args:
        tensor_protos: All tensors in ONNX model to apply external data to.
        tensor_dict: Dictionary of external data to apply to ONNX model.

    Returns:
        Names of tensors that were applied.
    """
    applied = set()
    for tensor in tensor_protos:
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
        set_external_data_flag(tensor, False)
        applied.add(name)

    return applied
