# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import Callable, Iterable

import onnx


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


def _get_initializer_tensors_from_graph(
    model_proto_graph: onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of initializer tensors from ONNX model graph."""
    yield from model_proto_graph.initializer
    for node in model_proto_graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(
                attribute, _get_initializer_tensors_from_graph
            )


def get_initializer_tensors(
    model_proto: onnx.ModelProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of initializer tensors from ONNX model."""
    yield from _get_initializer_tensors_from_graph(model_proto.graph)


def _get_attribute_tensors_from_graph(
    model_proto_graph: onnx.GraphProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model graph."""
    for node in model_proto_graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(
                attribute, _get_attribute_tensors_from_graph
            )


def _get_attribute_tensors(
    model_proto: onnx.ModelProto,
) -> Iterable[onnx.TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model."""
    yield from _get_attribute_tensors_from_graph(model_proto.graph)


def get_all_tensors(model_proto: onnx.ModelProto) -> Iterable[onnx.TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return itertools.chain(
        get_initializer_tensors(model_proto),
        _get_attribute_tensors(model_proto),
    )


def set_external_data_flag(tensor: onnx.TensorProto, flag: bool) -> None:
    """Set or unset the external data flag of a tensor."""
    # We do not need the metadata about external data
    if tensor.HasField("external_data"):
        tensor.ClearField("external_data")
    if flag:
        # After loading raw_data from external_data, change the state of tensors
        tensor.data_location = onnx.TensorProto.EXTERNAL
    else:
        tensor.data_location = onnx.TensorProto.DEFAULT


def clear_raw_data(tensor: onnx.TensorProto):
    """Clear raw_data of a tensor."""
    if tensor.HasField("raw_data"):
        tensor.ClearField("raw_data")
