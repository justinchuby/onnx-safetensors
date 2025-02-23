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


def get_initializers(
    graph: ir.Graph,
) -> Iterable[tuple[str, ir.TensorProtocol, Callable[[ir.TensorProtocol], None]]]:
    for value in graph.initializers.values():
        assert value.name is not None
        assert value.const_value is not None

        def _set_initializer(tensor: ir.TensorProtocol, v=value) -> None:
            v.const_value = tensor

        yield value.name, value.const_value, _set_initializer
    for node in graph:
        for attr in node.attributes.values():
            assert not isinstance(attr, ir.RefAttr)
            if attr.type == ir.AttributeType.GRAPH:
                yield from get_initializers(attr.as_graph())
            if attr.type == ir.AttributeType.GRAPHS:
                for g in attr.as_graphs():
                    yield from get_initializers(g)


def get_attributes(
    graph: ir.Graph,
) -> Iterable[tuple[str, ir.TensorProtocol, Callable[[ir.TensorProtocol], None]]]:
    """Get all tensors from ONNX model attributes."""
    for node in ir.traversal.RecursiveGraphIterator(graph):
        for name, attribute in node.attributes.items():
            assert not isinstance(attribute, ir.RefAttr)
            if attribute.type == ir.AttributeType.TENSOR:

                def _set_tensor(
                    tensor: ir.TensorProtocol, node=node, name=name
                ) -> None:
                    node.attributes[name] = ir.AttrTensor(name, tensor)

                yield name, attribute.as_tensor(), _set_tensor
