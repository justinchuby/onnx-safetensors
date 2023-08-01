from __future__ import annotations

import itertools
import os
import sys
from typing import Callable, Iterable

import onnx
import safetensors
import safetensors.numpy


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


def _get_initializer_tensors(
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


def _get_all_tensors(model_proto: onnx.ModelProto) -> Iterable[onnx.TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return itertools.chain(
        _get_initializer_tensors(model_proto),
        _get_attribute_tensors(model_proto),
    )


def _set_external_data_flag(tensor: onnx.TensorProto, flag: bool) -> None:
    # We do not need the metadata about external data
    del tensor.external_data[:]
    if flag:
        # After loading raw_data from external_data, change the state of tensors
        tensor.data_location = onnx.TensorProto.EXTERNAL
    else:
        tensor.data_location = onnx.TensorProto.DEFAULT
    return


def load_safetensors_file(
    model_proto: onnx.ModelProto, tensor_file: str | os.PathLike
) -> None:
    with safetensors.safe_open(tensor_file, "numpy") as f:
        keys = f.keys()
        for tensor in _get_all_tensors(model_proto):
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
            _set_external_data_flag(tensor, False)


def load_safetensors(model_proto: onnx.ModelProto, data: bytes) -> None:
    tensor_dict = safetensors.numpy.load(data)

    for tensor in _get_all_tensors(model_proto):
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
        _set_external_data_flag(tensor, False)


def save_safetensors(
    model_proto: onnx.ModelProto,
    tensor_file: str | os.PathLike,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    if convert_attribute:
        tensors = _get_all_tensors(model_proto)
    else:
        tensors = _get_initializer_tensors(model_proto)

    tensor_dict = {}

    for tensor in tensors:
        name = tensor.name
        if not (
            tensor.HasField("raw_data")
            and sys.getsizeof(tensor.raw_data) >= size_threshold
        ):
            continue
        tensor_dict[name] = onnx.helper.tensor_to_numpy(tensor)
        _set_external_data_flag(tensor, True)

    safetensors.numpy.save_file(tensor_dict, tensor_file)
